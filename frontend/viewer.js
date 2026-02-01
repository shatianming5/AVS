function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: ${id}`);
  return el;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function parseQuery() {
  const q = new URLSearchParams(window.location.search);
  return {
    page: parseInt(q.get("page") || "1", 10),
    pack: q.get("pack"),
    e: q.get("e"),
    token: q.get("token"),
  };
}

function pdfIdFromPath() {
  const m = window.location.pathname.match(/\/viewer\/(.+)$/);
  if (!m) throw new Error("Bad viewer path");
  return m[1];
}

async function fetchJson(url, options) {
  const res = await fetch(url, options || {});
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function setPage(pdfId, page, packId, evidenceId, token) {
  const q = new URLSearchParams();
  q.set("page", String(page));
  if (packId) q.set("pack", packId);
  if (evidenceId) q.set("e", evidenceId);
  if (token) q.set("token", token);
  window.location.search = q.toString();
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function normalizeBox(bb) {
  if (!Array.isArray(bb) || bb.length !== 4) return null;
  const x0 = Number(bb[0]);
  const y0 = Number(bb[1]);
  const x1 = Number(bb[2]);
  const y1 = Number(bb[3]);
  if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) return null;
  const a0 = clamp01(Math.min(x0, x1));
  const a1 = clamp01(Math.max(x0, x1));
  const b0 = clamp01(Math.min(y0, y1));
  const b1 = clamp01(Math.max(y0, y1));
  return [a0, b0, a1, b1];
}

function moveTooltip(tooltipEl, x, y) {
  const pad = 14;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const rect = tooltipEl.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + rect.width + pad > vw) left = Math.max(pad, x - rect.width - 14);
  if (top + rect.height + pad > vh) top = Math.max(pad, y - rect.height - 14);
  tooltipEl.style.left = `${Math.round(left)}px`;
  tooltipEl.style.top = `${Math.round(top)}px`;
}

function buildReferrersByEvidenceId(packId, packObj, token) {
  const map = {};
  const tokenQ = token ? `?token=${encodeURIComponent(token)}` : "";
  function href(anchor) {
    return `/pack/${packId}${tokenQ}${anchor || ""}`;
  }
  function push(evId, ref) {
    if (!evId) return;
    if (!map[evId]) map[evId] = [];
    map[evId].push(ref);
  }
  function addList(list, refBase) {
    for (const evId of list || []) push(evId, refBase);
  }

  const bp = (packObj && packObj.intro_blueprint) || {};
  const para = bp.paragraph_plan || [];
  for (const p of para) {
    const idx = p.paragraph_index;
    const lab = p.label || "Other";
    const title = `P${idx} · ${lab}`;
    addList(p.supporting_evidence || [], {
      ref_type: "paragraph_plan",
      id: String(idx),
      title,
      anchor: `#para_${idx}`,
      href: href(`#para_${idx}`),
    });
  }

  function addRules(rules, refType) {
    for (const r of rules || []) {
      const rid = r.rule_id;
      const title = r.title || "(untitled)";
      const anchor = rid ? `#blueprint_rule_${rid}` : "";
      addList(r.supporting_evidence || [], {
        ref_type: refType,
        id: String(rid || ""),
        title,
        anchor,
        href: href(anchor),
      });
    }
  }
  addRules(bp.story_rules || [], "story_rule");
  addRules(bp.claim_rules || [], "claim_rule");
  addRules(bp.checklist || [], "checklist");

  for (const t of packObj.templates || []) {
    const tid = t.template_id;
    const title = t.template_type || "Template";
    const anchor = tid ? `#template_${tid}` : "";
    addList(t.supporting_evidence || [], {
      ref_type: "template",
      id: String(tid || ""),
      title,
      anchor,
      href: href(anchor),
    });
  }

  for (const s of packObj.storyboard || []) {
    const sid = s.item_id;
    const title = s.figure_role || "Storyboard";
    const anchor = sid ? `#storyboard_${sid}` : "";
    addList(s.supporting_evidence || [], {
      ref_type: "storyboard",
      id: String(sid || ""),
      title,
      anchor,
      href: href(anchor),
    });
  }

  return map;
}

function renderEvidenceTooltipHtml(evObj, referrers) {
  if (!evObj) {
    return `<div class="tt-muted">No evidence selected.</div>`;
  }
  const usedBy =
    referrers && referrers.length
      ? `<div class="tt-usedby"><div class="tt-muted">Used by</div><ul>${referrers
          .slice(0, 12)
          .map((r) => `<li><a class="tt-link" href="${escapeHtml(r.href)}">${escapeHtml(r.title)}</a></li>`)
          .join("")}</ul></div>`
      : `<div class="tt-usedby"><div class="tt-muted">Used by</div><div>(none)</div></div>`;

  function kv(k, v) {
    return `<div class="tt-k">${escapeHtml(k)}</div><div class="tt-v">${escapeHtml(v)}</div>`;
  }

  const bbox = evObj.bbox_norm_list ? JSON.stringify(evObj.bbox_norm_list) : evObj.bbox_norm ? JSON.stringify(evObj.bbox_norm) : "";
  const excerpt = evObj.excerpt ? `<div class="tt-muted" style="margin-top:10px">excerpt</div><div>${escapeHtml(evObj.excerpt)}</div>` : "";
  const reason = evObj.reason ? `<div class="tt-muted" style="margin-top:10px">reason</div><div>${escapeHtml(evObj.reason)}</div>` : "";

  return (
    usedBy +
    `<div class="tt-kv">` +
    kv("pdf_id", evObj.pdf_id || "") +
    kv("page_index", String(evObj.page_index || "")) +
    kv("evidence_id", evObj.evidence_id || "") +
    kv("block_id", evObj.block_id || "") +
    kv("kind", evObj.kind || "") +
    kv("confidence", String(evObj.confidence)) +
    (bbox ? kv("bbox_norm", bbox) : kv("bbox_norm", "(none)")) +
    `</div>` +
    excerpt +
    reason
  );
}

function renderAnnotationTooltipHtml(ann) {
  if (!ann) return `<div class="tt-muted">Missing annotation.</div>`;
  function kv(k, v) {
    return `<div class="tt-k">${escapeHtml(k)}</div><div class="tt-v">${escapeHtml(v)}</div>`;
  }
  const bbox = ann.bbox_norm ? JSON.stringify(ann.bbox_norm) : "";
  const note = ann.note ? `<div class="tt-muted" style="margin-top:10px">note</div><div>${escapeHtml(ann.note)}</div>` : "";
  const skill = ann.skill_title ? `<div class="tt-muted" style="margin-top:10px">skill</div><div>${escapeHtml(ann.skill_title)}</div>` : "";
  return (
    `<h3 style="margin:0 0 6px 0">Annotation</h3>` +
    `<div class="tt-kv">` +
    kv("pdf_id", ann.pdf_id || "") +
    kv("page_index", String(ann.page_index || "")) +
    kv("annotation_id", ann.annotation_id || "") +
    (bbox ? kv("bbox_norm", bbox) : kv("bbox_norm", "(none)")) +
    `</div>` +
    skill +
    note
  );
}

function flattenSkills(packObj) {
  const out = [];
  const bp = (packObj && packObj.intro_blueprint) || {};
  for (const p of bp.paragraph_plan || []) {
    const idx = p.paragraph_index;
    const lab = p.label || "Other";
    out.push({ type: "paragraph_plan", id: String(idx), title: `P${idx} · ${lab}` });
  }
  function addRules(rules, typ) {
    for (const r of rules || []) {
      out.push({ type: typ, id: String(r.rule_id || ""), title: String(r.title || "(untitled)") });
    }
  }
  addRules(bp.story_rules || [], "story_rule");
  addRules(bp.claim_rules || [], "claim_rule");
  addRules(bp.checklist || [], "checklist");
  for (const t of packObj.templates || []) out.push({ type: "template", id: String(t.template_id || ""), title: `Template: ${t.template_type}` });
  for (const s of packObj.storyboard || []) out.push({ type: "storyboard", id: String(s.item_id || ""), title: `Storyboard: ${s.figure_role}` });
  return out.filter((x) => x.id);
}

window.addEventListener("DOMContentLoaded", async () => {
  const pdfId = pdfIdFromPath();
  const { page, pack, e, token } = parseQuery();
  const access = token || localStorage.getItem("paper_skill_owner_token");

  if (!pack || !access) {
    byId("ev").textContent = "Missing pack/token.";
    return;
  }
  if (token && !localStorage.getItem("paper_skill_owner_token")) {
    localStorage.setItem("paper_skill_owner_token", token);
  }

  const pdfjsLib = window.pdfjsLib;
  if (!pdfjsLib) {
    byId("ev").textContent = "Missing pdfjsLib (PDF.js not loaded).";
    return;
  }
  if (pdfjsLib.GlobalWorkerOptions) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.js";
  }

  const tooltip = byId("tooltip");
  let hideTimer = null;
  function showTooltip(html, x, y) {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    tooltip.innerHTML = html;
    tooltip.classList.remove("hidden");
    moveTooltip(tooltip, x, y);
  }
  function scheduleHideTooltip() {
    if (hideTimer) return;
    hideTimer = setTimeout(() => {
      tooltip.classList.add("hidden");
      hideTimer = null;
    }, 160);
  }
  tooltip.addEventListener("mouseenter", () => {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  });
  tooltip.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));

  const headers = access ? { "X-Pack-Token": access } : {};

  let packObj = null;
  let evidenceIndex = null;
  let annotations = [];
  try {
    const [p, ev, ann] = await Promise.all([
      fetchJson(`/api/skillpacks/${pack}`, { headers }),
      fetchJson(`/api/evidence/${pack}`, { headers }),
      fetchJson(`/api/annotations?pack_id=${encodeURIComponent(pack)}&pdf_id=${encodeURIComponent(pdfId)}`, { headers }),
    ]);
    packObj = p;
    evidenceIndex = ev;
    annotations = (ann && ann.annotations) || [];
  } catch (err) {
    byId("ev").textContent = `Error loading pack/evidence/annotations: ${err.message}`;
    return;
  }

  const evidenceById = {};
  for (const ev of evidenceIndex.evidence || []) evidenceById[ev.evidence_id] = ev;
  const referrersByEvidenceId = buildReferrersByEvidenceId(pack, packObj, access);

  const evSelected = e ? evidenceById[e] || null : null;
  const refSelected = (evSelected && referrersByEvidenceId[evSelected.evidence_id]) || [];
  byId("usedby").innerHTML =
    refSelected && refSelected.length
      ? `Used by: ${refSelected
          .slice(0, 6)
          .map((r) => `<a class="link" href="${escapeHtml(r.href)}">${escapeHtml(r.title)}</a>`)
          .join(" · ")}`
      : "Used by: (none)";
  byId("ev").textContent = evSelected ? JSON.stringify(evSelected, null, 2) : "No evidence selected.";

  // PDF load
  const pdfUrl = `/api/skillpacks/${pack}/pdfs/${pdfId}/file?token=${encodeURIComponent(access)}`;
  const loadingTask = pdfjsLib.getDocument(pdfUrl);
  const pdfDoc = await loadingTask.promise;
  const numPages = pdfDoc.numPages || 0;

  const safePage = Math.max(1, Math.min(numPages || 1, Number.isFinite(page) ? page : 1));
  if (safePage !== page) {
    setPage(pdfId, safePage, pack, e, access);
    return;
  }

  byId("download").href = pdfUrl;
  byId("page_input").value = String(safePage);
  byId("page_total").textContent = numPages ? `/ ${numPages}` : "";

  function updateNav() {
    byId("prev").disabled = safePage <= 1;
    byId("next").disabled = numPages ? safePage >= numPages : false;
  }
  updateNav();
  byId("prev").addEventListener("click", () => setPage(pdfId, Math.max(1, safePage - 1), pack, e, access));
  byId("next").addEventListener("click", () => setPage(pdfId, safePage + 1, pack, e, access));
  byId("page_input").addEventListener("keydown", (evt) => {
    if (evt.key !== "Enter") return;
    const v = parseInt(byId("page_input").value || "1", 10);
    if (Number.isFinite(v)) setPage(pdfId, v, pack, e, access);
  });

  let scale = 1.2;
  function updateZoomLabel() {
    byId("zoom_reset").textContent = `${Math.round(scale * 100)}%`;
  }

  const canvas = byId("canvas");
  const wrap = byId("wrap");
  const overlay = byId("overlay");

  function drawBoxPx({ cls, left, top, width, height, onEnter, onMove, onLeave }) {
    const div = document.createElement("div");
    div.className = cls;
    div.style.left = `${left}px`;
    div.style.top = `${top}px`;
    div.style.width = `${width}px`;
    div.style.height = `${height}px`;
    if (onEnter) div.addEventListener("mouseenter", onEnter);
    if (onMove) div.addEventListener("mousemove", onMove);
    if (onLeave) div.addEventListener("mouseleave", onLeave);
    overlay.appendChild(div);
    return div;
  }

  function renderOverlays(viewW, viewH) {
    overlay.innerHTML = "";
    const evidenceForPage = (evidenceIndex.evidence || []).filter((x) => x.pdf_id === pdfId && Number(x.page_index) === safePage);
    const annForPage = (annotations || []).filter((a) => a.pdf_id === pdfId && Number(a.page_index) === safePage);

    for (const ev of evidenceForPage) {
      const ref = referrersByEvidenceId[ev.evidence_id] || [];
      const boxes = Array.isArray(ev.bbox_norm_list) && ev.bbox_norm_list.length ? ev.bbox_norm_list : ev.bbox_norm ? [ev.bbox_norm] : [];
      for (const bb of boxes) {
        const b = normalizeBox(bb);
        if (!b) continue;
        const [x0, y0, x1, y1] = b;
        drawBoxPx({
          cls: `hl${evSelected && evSelected.evidence_id === ev.evidence_id ? " selected" : ""}`,
          left: x0 * viewW,
          top: y0 * viewH,
          width: (x1 - x0) * viewW,
          height: (y1 - y0) * viewH,
          onEnter: (evt) => showTooltip(renderEvidenceTooltipHtml(ev, ref), evt.clientX, evt.clientY),
          onMove: (evt) => !tooltip.classList.contains("hidden") && moveTooltip(tooltip, evt.clientX, evt.clientY),
          onLeave: () => scheduleHideTooltip(),
        });
      }
    }

    for (const ann of annForPage) {
      const b = normalizeBox(ann.bbox_norm);
      if (!b) continue;
      const [x0, y0, x1, y1] = b;
      drawBoxPx({
        cls: "ann",
        left: x0 * viewW,
        top: y0 * viewH,
        width: (x1 - x0) * viewW,
        height: (y1 - y0) * viewH,
        onEnter: (evt) => showTooltip(renderAnnotationTooltipHtml(ann), evt.clientX, evt.clientY),
        onMove: (evt) => !tooltip.classList.contains("hidden") && moveTooltip(tooltip, evt.clientX, evt.clientY),
        onLeave: () => scheduleHideTooltip(),
      });
    }
  }

  function renderAnnList() {
    const annForPage = (annotations || []).filter((a) => a.pdf_id === pdfId && Number(a.page_index) === safePage);
    if (!annForPage.length) {
      byId("ann_list").innerHTML = `<div class="subtle">No annotations on this page.</div>`;
      return;
    }
    byId("ann_list").innerHTML = annForPage
      .map((a) => {
        const title = a.skill_title ? `<div><span class="badge">${escapeHtml(a.skill_title)}</span></div>` : "";
        const note = a.note ? `<div class="subtle" style="margin-top:6px">${escapeHtml(a.note)}</div>` : "";
        return `<div style="margin-bottom:12px">
          <div class="row" style="justify-content: space-between">
            <div class="subtle">id: ${escapeHtml(String(a.annotation_id || "").slice(0, 10))}</div>
            <button data-del="${escapeHtml(a.annotation_id || "")}" style="padding:6px 10px">Delete</button>
          </div>
          ${title}
          ${note}
        </div>`;
      })
      .join("");
  }

  byId("ann_list").addEventListener("click", async (evt) => {
    const btn = evt.target && evt.target.closest ? evt.target.closest("button[data-del]") : null;
    if (!btn) return;
    const annId = btn.getAttribute("data-del");
    if (!annId) return;
    try {
      const res = await fetch(`/api/annotations/${encodeURIComponent(annId)}`, {
        method: "DELETE",
        headers,
      });
      if (!res.ok) throw new Error(await res.text());
      annotations = (annotations || []).filter((a) => a.annotation_id !== annId);
      renderAnnList();
      // Re-render overlays at current canvas size.
      const rect = canvas.getBoundingClientRect();
      renderOverlays(rect.width, rect.height);
    } catch (err) {
      byId("ann_hint").textContent = `Delete failed: ${err.message}`;
    }
  });

  async function renderPdfPage() {
    byId("meta").textContent = `pdf_id=${pdfId} | page=${safePage}${numPages ? `/${numPages}` : ""}`;
    const pageObj = await pdfDoc.getPage(safePage);
    const viewport = pageObj.getViewport({ scale });
    const ctx = canvas.getContext("2d");
    canvas.width = Math.floor(viewport.width);
    canvas.height = Math.floor(viewport.height);
    canvas.style.width = `${Math.floor(viewport.width)}px`;
    canvas.style.height = `${Math.floor(viewport.height)}px`;
    wrap.style.width = canvas.style.width;
    wrap.style.height = canvas.style.height;

    const renderTask = pageObj.render({ canvasContext: ctx, viewport });
    await renderTask.promise;

    const rect = canvas.getBoundingClientRect();
    renderOverlays(rect.width, rect.height);
    renderAnnList();
    updateZoomLabel();
  }

  updateZoomLabel();
  byId("zoom_out").addEventListener("click", async () => {
    scale = Math.max(0.5, scale / 1.15);
    await renderPdfPage();
  });
  byId("zoom_in").addEventListener("click", async () => {
    scale = Math.min(3.0, scale * 1.15);
    await renderPdfPage();
  });
  byId("zoom_reset").addEventListener("click", async () => {
    scale = 1.2;
    await renderPdfPage();
  });

  // Annotation creation
  const annDialog = byId("ann_dialog");
  const annSkill = byId("ann_skill");
  const annNote = byId("ann_note");
  const annotateBtn = byId("annotate_toggle");
  let annotateOn = false;
  let pendingBox = null;
  let draftDiv = null;

  function setAnnotate(on) {
    annotateOn = !!on;
    annotateBtn.classList.toggle("active", annotateOn);
    byId("ann_hint").textContent = annotateOn ? "Drag on the page to create an annotation (owner only)." : "";
  }
  annotateBtn.addEventListener("click", () => setAnnotate(!annotateOn));

  function fillSkillOptions() {
    const skills = flattenSkills(packObj);
    annSkill.innerHTML = skills
      .map((s) => `<option value="${escapeHtml(s.type)}:${escapeHtml(s.id)}">${escapeHtml(s.title)}</option>`)
      .join("");
  }
  fillSkillOptions();

  function removeDraft() {
    if (draftDiv && draftDiv.parentNode) draftDiv.parentNode.removeChild(draftDiv);
    draftDiv = null;
  }

  function showAnnDialog() {
    annNote.value = "";
    if (annSkill.options.length) annSkill.selectedIndex = 0;
    annDialog.showModal();
  }

  byId("ann_cancel").addEventListener("click", () => {
    removeDraft();
    pendingBox = null;
    annDialog.close();
  });

  byId("ann_form").addEventListener("submit", async (evt) => {
    evt.preventDefault();
    if (!pendingBox) return;
    const selected = String(annSkill.value || "");
    const [skillRefType, skillRefId] = selected.includes(":") ? selected.split(":", 2) : ["", ""];
    const skillTitle = annSkill.options[annSkill.selectedIndex] ? annSkill.options[annSkill.selectedIndex].textContent : "";
    try {
      const res = await fetch(`/api/annotations`, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          pack_id: pack,
          pdf_id: pdfId,
          page_index: safePage,
          bbox_norm: pendingBox,
          note: annNote.value || null,
          skill_ref_type: skillRefType || null,
          skill_ref_id: skillRefId || null,
          skill_title: skillTitle || null,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const ann = data.annotation;
      annotations = (annotations || []).concat([ann]);
      pendingBox = null;
      removeDraft();
      annDialog.close();
      const rect = canvas.getBoundingClientRect();
      renderOverlays(rect.width, rect.height);
      renderAnnList();
    } catch (err) {
      byId("ann_hint").textContent = `Create failed: ${err.message}`;
    }
  });

  function onDragStart(evt) {
    if (!annotateOn) return;
    if (evt.button !== 0) return;
    evt.preventDefault();
    const rect = overlay.getBoundingClientRect();
    const sx = evt.clientX - rect.left;
    const sy = evt.clientY - rect.top;
    const div = document.createElement("div");
    div.className = "ann draft";
    div.style.left = `${sx}px`;
    div.style.top = `${sy}px`;
    div.style.width = `1px`;
    div.style.height = `1px`;
    overlay.appendChild(div);
    draftDiv = div;

    function onMove(e2) {
      const x = e2.clientX - rect.left;
      const y = e2.clientY - rect.top;
      const x0 = Math.min(sx, x);
      const y0 = Math.min(sy, y);
      const x1 = Math.max(sx, x);
      const y1 = Math.max(sy, y);
      div.style.left = `${x0}px`;
      div.style.top = `${y0}px`;
      div.style.width = `${Math.max(1, x1 - x0)}px`;
      div.style.height = `${Math.max(1, y1 - y0)}px`;
    }
    function onUp(e2) {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      const x = e2.clientX - rect.left;
      const y = e2.clientY - rect.top;
      const x0 = Math.min(sx, x);
      const y0 = Math.min(sy, y);
      const x1 = Math.max(sx, x);
      const y1 = Math.max(sy, y);
      if (Math.abs(x1 - x0) < 6 || Math.abs(y1 - y0) < 6) {
        removeDraft();
        pendingBox = null;
        return;
      }
      pendingBox = normalizeBox([x0 / rect.width, y0 / rect.height, x1 / rect.width, y1 / rect.height]);
      if (!pendingBox) {
        removeDraft();
        return;
      }
      showAnnDialog();
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  overlay.addEventListener("mousedown", onDragStart);

  await renderPdfPage();
});
