from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

# NOTE: Keep transformer model imports local to avoid making the whole repo depend
# on a specific Transformers version unless the user actually runs VLM evals.


@dataclass(frozen=True)
class QwenVLConfig:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda:0"
    dtype: str = "bfloat16"  # float16|bfloat16|float32
    attn_implementation: str | None = None  # e.g. "flash_attention_2" if installed
    min_pixels: int | None = None
    max_pixels: int | None = None


def _torch_dtype(name: str) -> torch.dtype:
    s = str(name)
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype={name!r}; expected float16/bfloat16/float32")


def _first_param_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _load_images(paths: list[Path]) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def build_mcq_prompt(*, question: str, options: list[str]) -> str:
    letters = ["A", "B", "C", "D", "E"]
    opts = []
    for i, o in enumerate(options):
        if i >= len(letters):
            break
        opts.append(f"{letters[i]}. {str(o)}")
    opts_txt = "\n".join(opts)
    q = str(question).strip()
    return (
        "You are given a sequence of video frames (in chronological order). "
        "Answer the multiple-choice question by outputting a single letter only.\n\n"
        f"Question: {q}\n\n"
        f"Options:\n{opts_txt}\n\n"
        "Answer (single letter):"
    )


_ANS_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def parse_mcq_answer(text: str, *, num_options: int = 5) -> int | None:
    letters = ["A", "B", "C", "D", "E"][: int(num_options)]
    m = _ANS_RE.search(str(text))
    if not m:
        return None
    c = str(m.group(1)).upper()
    if c not in letters:
        return None
    return int(letters.index(c))


@dataclass(frozen=True)
class VLMAnswer:
    ok: bool
    pred_idx: int | None
    raw_text: str
    timings: dict


class QwenVL:
    def __init__(self, cfg: QwenVLConfig = QwenVLConfig()):
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Qwen2-VL requires a recent `transformers` with Qwen2VLForConditionalGeneration. "
                "Try: pip install -U transformers"
            ) from e

        self.cfg = cfg
        self.dtype = _torch_dtype(cfg.dtype)

        kwargs = {}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = str(cfg.attn_implementation)

        self.processor = AutoProcessor.from_pretrained(str(cfg.model_name))
        if cfg.min_pixels is not None:
            self.processor.image_processor.min_pixels = int(cfg.min_pixels)
        if cfg.max_pixels is not None:
            self.processor.image_processor.max_pixels = int(cfg.max_pixels)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(cfg.model_name),
            torch_dtype=self.dtype,
            device_map=None,
            **kwargs,
        )
        self.model.eval()

        dev = torch.device(str(cfg.device))
        self.model.to(dev)

    def _build_inputs(self, *, image_paths: list[Path], prompt: str) -> dict:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as e:  # noqa: BLE001
            raise ImportError("missing dependency: qwen-vl-utils (pip install qwen-vl-utils)") from e

        images = _load_images(image_paths)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": str(prompt)}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        dev = _first_param_device(self.model)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def answer_mcq_generate(
        self,
        *,
        image_paths: list[Path],
        question: str,
        options: list[str],
        max_new_tokens: int = 32,
    ) -> VLMAnswer:
        t0 = time.time()
        prompt = build_mcq_prompt(question=question, options=options)
        inputs = self._build_inputs(image_paths=image_paths, prompt=prompt)
        t1 = time.time()

        out = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=int(max_new_tokens),
        )
        t2 = time.time()

        # Slice off the prompt tokens.
        in_len = int(inputs["input_ids"].shape[1])
        gen_ids = out[:, in_len:]
        raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred = parse_mcq_answer(raw, num_options=len(options))
        timings = {
            "prepare_s": float(t1 - t0),
            "generate_s": float(t2 - t1),
            "total_s": float(t2 - t0),
            "num_images": int(len(image_paths)),
            "max_new_tokens": int(max_new_tokens),
        }
        return VLMAnswer(ok=pred is not None, pred_idx=pred, raw_text=str(raw), timings=timings)

    @torch.no_grad()
    def answer_mcq_ppl(
        self,
        *,
        image_paths: list[Path],
        question: str,
        options: list[str],
        candidates: list[str] | None = None,
    ) -> VLMAnswer:
        """
        Multiple-choice selection by scoring candidate single-letter answers with teacher forcing.

        Notes:
          - This is more stable than free-form generation, but assumes the model follows the "single letter" format.
        """
        t0 = time.time()
        prompt = build_mcq_prompt(question=question, options=options)
        inputs = self._build_inputs(image_paths=image_paths, prompt=prompt)
        t1 = time.time()

        tok = self.processor.tokenizer
        if candidates is None:
            letters = ["A", "B", "C", "D", "E"][: len(options)]
            candidates = letters
        cand_ids = [tok(str(c), add_special_tokens=False).input_ids for c in candidates]

        # Base prompt tokens.
        base_ids = inputs["input_ids"]
        base_attn = inputs["attention_mask"]
        base_len = int(base_ids.shape[1])

        # Score each candidate by log P(cand | prompt, images).
        scores: list[float] = []
        dev = _first_param_device(self.model)

        for ids in cand_ids:
            if not ids:
                scores.append(float("-inf"))
                continue
            extra = torch.tensor([ids], dtype=base_ids.dtype, device=dev)
            full_ids = torch.cat([base_ids, extra], dim=1)
            full_attn = torch.cat([base_attn, torch.ones_like(extra, device=dev)], dim=1)

            out = self.model(input_ids=full_ids, attention_mask=full_attn, **{k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")})
            logits = out.logits  # [1, L, V]
            logp = torch.log_softmax(logits, dim=-1)

            lp = 0.0
            # token at position t is predicted by logits at t-1
            for j, tid in enumerate(ids):
                pos = base_len + j
                lp += float(logp[0, pos - 1, int(tid)].item())
            scores.append(float(lp))

        best = int(torch.tensor(scores).argmax().item())
        pred_text = str(candidates[best])
        pred = parse_mcq_answer(pred_text, num_options=len(options))
        t2 = time.time()

        timings = {
            "prepare_s": float(t1 - t0),
            "score_s": float(t2 - t1),
            "total_s": float(t2 - t0),
            "num_images": int(len(image_paths)),
        }
        ok = pred is not None and float(scores[best]) != float("-inf")
        raw = f"ppl_best={pred_text} scores={scores}"
        return VLMAnswer(ok=ok, pred_idx=pred, raw_text=raw, timings=timings)
