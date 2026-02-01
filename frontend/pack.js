function packIdFromPath() {
  const m = window.location.pathname.match(/\/pack\/(.+)$/);
  if (!m) throw new Error("Bad pack path");
  return m[1];
}

function accessToken() {
  const q = new URLSearchParams(window.location.search);
  const t = q.get("token");
  if (t) return t;
  return localStorage.getItem("paper_skill_owner_token");
}

async function fetchJson(url, token) {
  const res = await fetch(url, {
    headers: token ? { "X-Pack-Token": token } : {},
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function evidenceChip(packId, ev, { token, pdfLabel, skillTitle }) {
  const page = ev.page_index;
  const pdfId = ev.pdf_id;
  const eid = ev.evidence_id;
  const tokenQ = token ? `&token=${encodeURIComponent(token)}` : "";
  const label = `${pdfLabel(pdfId)} p.${page}`;
  return `<a class="ev-chip" data-ev="${escapeHtml(eid || "")}" data-skill="${escapeHtml(
    skillTitle || ""
  )}" href="/viewer/${pdfId}?page=${page}&pack=${packId}&e=${eid}${tokenQ}">${escapeHtml(label)}</a>`;
}

function renderEvidenceChips(packId, evidence, { token, pdfLabel, skillTitle }) {
  if (!evidence || evidence.length === 0) return `<div class="subtle">No evidence found.</div>`;
  return `<div class="ev-chips">${evidence
    .map((ev) => evidenceChip(packId, ev, { token, pdfLabel, skillTitle }))
    .join("")}</div>`;
}

function section(title, innerHtml) {
  return `<section class="section"><h2>${escapeHtml(title)}</h2>${innerHtml}</section>`;
}

function badge(text) {
  if (!text) return "";
  return `<span class="badge">${escapeHtml(text)}</span>`;
}

function itemCard({ id, title, metaHtml, bodyHtml, evidenceHtml }) {
  const idAttr = id ? ` id="${escapeHtml(id)}"` : "";
  const meta = metaHtml ? `<div class="item-meta">${metaHtml}</div>` : "";
  const body = bodyHtml ? `<div class="item-desc">${bodyHtml}</div>` : "";
  const evidence = evidenceHtml || "";
  return `<div class="item"${idAttr}>
    <div class="item-header">
      <div class="item-title">${escapeHtml(title)}</div>
      ${meta}
    </div>
    ${body}
    ${evidence}
  </div>`;
}

function ruleItem(packId, r, { token, pdfLabel, groupLabel }) {
  const rid = r.rule_id ? `blueprint_rule_${r.rule_id}` : "";
  const meta = badge(groupLabel);
  const body = escapeHtml(r.description || "");
  const ev = renderEvidenceChips(packId, r.supporting_evidence, { token, pdfLabel, skillTitle: r.title });
  return itemCard({ id: rid, title: r.title || "(untitled)", metaHtml: meta, bodyHtml: body, evidenceHtml: ev });
}

function templateItem(packId, t, { token, pdfLabel }) {
  const slots = t.slot_schema ? escapeHtml(JSON.stringify(t.slot_schema, null, 2)) : "";
  const tid = t.template_id ? `template_${t.template_id}` : "";
  const meta = badge("Template");
  const body =
    `<pre class="pre">${escapeHtml(t.text_with_slots || "")}</pre>` +
    (slots
      ? `<details style="margin-top:10px"><summary class="subtle">slot_schema</summary><pre class="pre">${slots}</pre></details>`
      : "");
  const ev = renderEvidenceChips(packId, t.supporting_evidence, { token, pdfLabel, skillTitle: `Template: ${t.template_type}` });
  return itemCard({ id: tid, title: t.template_type || "Template", metaHtml: meta, bodyHtml: body, evidenceHtml: ev });
}

function storyboardItem(packId, s, { token, pdfLabel }) {
  const sid = s.item_id ? `storyboard_${s.item_id}` : "";
  const meta = badge("Storyboard");
  const title = `${s.figure_role || "Storyboard"} (recommended after: ${String(s.recommended_position || "")})`;
  const body = escapeHtml(s.caption_formula || "");
  const ev = renderEvidenceChips(packId, s.supporting_evidence, { token, pdfLabel, skillTitle: `Storyboard: ${s.figure_role}` });
  return itemCard({ id: sid, title, metaHtml: meta, bodyHtml: body, evidenceHtml: ev });
}

function weakItemsSection(pack) {
  const weak = (pack.quality_report && pack.quality_report.weak_items) || [];
  if (!weak || weak.length === 0) return "";
  function anchorFor(w) {
    if (w.kind === "blueprint_rule") return `#blueprint_rule_${w.id}`;
    if (w.kind === "template") return `#template_${w.id}`;
    if (w.kind === "storyboard") return `#storyboard_${w.id}`;
    return "";
  }
  const items = weak
    .map((w) => {
      const href = anchorFor(w);
      const title = `${w.kind}: ${w.title || w.id} (${w.reason || "issue"})`;
      if (href) return `<li><a class="link" href="${escapeHtml(href)}">${escapeHtml(title)}</a></li>`;
      return `<li>${escapeHtml(title)}</li>`;
    })
    .join("");
  return section("Weak items", `<div class="subtle">Items missing evidence or downgraded.</div><ul class="ev-list">${items}</ul>`);
}

window.addEventListener("DOMContentLoaded", async () => {
  const packId = packIdFromPath();
  const token = accessToken();
  if (!token) {
    document.getElementById("content").textContent = "Missing token. Open via a share link or build from this browser.";
    return;
  }
  if (token && !localStorage.getItem("paper_skill_owner_token")) {
    localStorage.setItem("paper_skill_owner_token", token);
  }

  const pack = await fetchJson(`/api/skillpacks/${packId}`, token);
  const evidence = await fetchJson(`/api/evidence/${packId}`, token);

  document.getElementById("title").textContent = `${pack.pack_name}`;
  document.getElementById("meta").textContent = `${pack.pack_id}`;

  // Download links (token via query param, since it's an <a>)
  document.getElementById("download_json").href = `/api/skillpacks/${packId}/download?format=json&token=${encodeURIComponent(token)}`;
  document.getElementById("download_yaml").href = `/api/skillpacks/${packId}/download?format=yaml&token=${encodeURIComponent(token)}`;

  // Share/unshare
  const shareBox = document.getElementById("share_box");
  document.getElementById("share_btn").addEventListener("click", async () => {
    const token = accessToken();
    const res = await fetch(`/api/skillpacks/${packId}/share`, {
      method: "POST",
      headers: { "X-Pack-Token": token || "" },
    });
    if (!res.ok) {
      shareBox.textContent = `Share failed: ${await res.text()}`;
      return;
    }
    const data = await res.json();
    shareBox.innerHTML = `Share link: <a class=\"link\" href=\"${data.share_url}\">${data.share_url}</a>`;
  });
  document.getElementById("unshare_btn").addEventListener("click", async () => {
    const token = accessToken();
    const res = await fetch(`/api/skillpacks/${packId}/unshare`, {
      method: "POST",
      headers: { "X-Pack-Token": token || "" },
    });
    if (!res.ok) {
      shareBox.textContent = `Unshare failed: ${await res.text()}`;
      return;
    }
    shareBox.textContent = "Unshared.";
  });

  // Merge evidence pointers by evidence_id onto objects inside pack for rendering convenience
  const evidenceById = {};
  for (const ev of evidence.evidence || []) evidenceById[ev.evidence_id] = ev;
  function pdfLabel(pdfId) {
    const ids = pack.pdf_ids || [];
    const idx = ids.indexOf(pdfId);
    if (idx >= 0) return `Paper ${idx + 1}`;
    return `PDF ${String(pdfId || "").slice(0, 8)}`;
  }

  const tooltip = document.getElementById("tooltip");
  function moveTooltip(x, y) {
    const pad = 14;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const rect = tooltip.getBoundingClientRect();
    let left = x + 14;
    let top = y + 14;
    if (left + rect.width + pad > vw) left = Math.max(pad, x - rect.width - 14);
    if (top + rect.height + pad > vh) top = Math.max(pad, y - rect.height - 14);
    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;
  }
  function renderEvidenceTooltipHtml(evObj, skillTitle) {
    if (!evObj) return `<div class="tt-muted">Missing evidence.</div>`;
    function kv(k, v) {
      return `<div class="tt-k">${escapeHtml(k)}</div><div class="tt-v">${escapeHtml(v)}</div>`;
    }
    const bbox = evObj.bbox_norm_list
      ? JSON.stringify(evObj.bbox_norm_list)
      : evObj.bbox_norm
        ? JSON.stringify(evObj.bbox_norm)
        : "";
    const excerpt = evObj.excerpt
      ? `<div class="tt-muted" style="margin-top:10px">excerpt</div><div>${escapeHtml(evObj.excerpt)}</div>`
      : "";
    const reason = evObj.reason
      ? `<div class="tt-muted" style="margin-top:10px">reason</div><div>${escapeHtml(evObj.reason)}</div>`
      : "";
    return (
      `<h3>${escapeHtml(skillTitle || "Skill")}</h3>` +
      `<div class="tt-muted">${escapeHtml(pdfLabel(evObj.pdf_id))} · p.${escapeHtml(String(evObj.page_index || ""))}</div>` +
      `<div class="tt-kv">` +
      kv("evidence_id", evObj.evidence_id || "") +
      kv("pdf_id", evObj.pdf_id || "") +
      kv("page_index", String(evObj.page_index || "")) +
      kv("block_id", evObj.block_id || "") +
      kv("kind", evObj.kind || "") +
      kv("confidence", String(evObj.confidence)) +
      (bbox ? kv("bbox_norm", bbox) : kv("bbox_norm", "(none)")) +
      `</div>` +
      excerpt +
      reason
    );
  }
  function showTooltip(html, x, y) {
    tooltip.innerHTML = html;
    tooltip.classList.remove("hidden");
    moveTooltip(x, y);
  }
  function hideTooltip() {
    tooltip.classList.add("hidden");
  }
  function inflate(list) {
    return (list || []).map((idOrObj) => {
      if (typeof idOrObj === "string") return evidenceById[idOrObj] || { evidence_id: idOrObj };
      if (idOrObj && typeof idOrObj === "object" && idOrObj.evidence_id) return evidenceById[idOrObj.evidence_id] || idOrObj;
      return idOrObj;
    });
  }

  const blueprint = pack.intro_blueprint || {};
  (blueprint.story_rules || []).forEach((r) => (r.supporting_evidence = inflate(r.supporting_evidence)));
  (blueprint.claim_rules || []).forEach((r) => (r.supporting_evidence = inflate(r.supporting_evidence)));
  (blueprint.checklist || []).forEach((r) => (r.supporting_evidence = inflate(r.supporting_evidence)));

  (pack.templates || []).forEach((t) => (t.supporting_evidence = inflate(t.supporting_evidence)));
  (pack.storyboard || []).forEach((s) => (s.supporting_evidence = inflate(s.supporting_evidence)));

  const content = document.getElementById("content");
  const parts = [];

  const weakHtml = weakItemsSection(pack);
  if (weakHtml) parts.push(weakHtml);

  parts.push(
    section(
      "Intro Blueprint",
      `<div class="subtle">Paragraph plan</div>` +
        (blueprint.paragraph_plan || [])
          .map((p) => {
            const idx = p.paragraph_index;
            const lab = p.label || "Other";
            const title = `P${idx} · ${lab}`;
            const ev = inflate(p.supporting_evidence || []);
            const evHtml = renderEvidenceChips(packId, ev, { token, pdfLabel, skillTitle: title });
            return itemCard({
              id: `para_${idx}`,
              title,
              metaHtml: badge("Blueprint"),
              bodyHtml: escapeHtml(p.description || ""),
              evidenceHtml: evHtml,
            });
          })
          .join("") +
        `<h3 style="margin-top:18px">Story rules</h3>` +
        (blueprint.story_rules || []).map((r) => ruleItem(packId, r, { token, pdfLabel, groupLabel: "Story rule" })).join("") +
        `<h3 style="margin-top:18px">Claim rules</h3>` +
        (blueprint.claim_rules || []).map((r) => ruleItem(packId, r, { token, pdfLabel, groupLabel: "Claim rule" })).join("") +
        `<h3 style="margin-top:18px">Checklist</h3>` +
        (blueprint.checklist || []).map((r) => ruleItem(packId, r, { token, pdfLabel, groupLabel: "Checklist" })).join("")
    )
  );

  parts.push(section("Templates", (pack.templates || []).map((t) => templateItem(packId, t, { token, pdfLabel })).join("")));

  parts.push(section("Figure storyboard", (pack.storyboard || []).map((s) => storyboardItem(packId, s, { token, pdfLabel })).join("")));

  const qr = pack.quality_report || {};
  const metrics = `<div class="ev-chips">
    <span class="badge">evidence_coverage: ${escapeHtml(String(qr.evidence_coverage))}</span>
    <span class="badge">structure_strength: ${escapeHtml(String(qr.structure_strength))}</span>
    <span class="badge">template_slot_score: ${escapeHtml(String(qr.template_slot_score))}</span>
    <span class="badge">plagiarism_max: ${escapeHtml(String(qr.plagiarism_max_similarity))}</span>
  </div>`;
  const qrJson = `<details style="margin-top:12px"><summary class="subtle">raw quality_report</summary><pre class="pre">${escapeHtml(
    JSON.stringify(qr, null, 2)
  )}</pre></details>`;
  parts.push(section("Quality report", metrics + qrJson));

  content.innerHTML = parts.join("");

  let activeChip = null;
  content.addEventListener("mouseover", (evt) => {
    const chip = evt.target && evt.target.closest ? evt.target.closest("a.ev-chip") : null;
    if (!chip) return;
    activeChip = chip;
    const evId = chip.getAttribute("data-ev");
    const skillTitle = chip.getAttribute("data-skill") || "";
    const evObj = evidenceById[evId] || null;
    showTooltip(renderEvidenceTooltipHtml(evObj, skillTitle), evt.clientX, evt.clientY);
  });
  content.addEventListener("mousemove", (evt) => {
    if (!activeChip || tooltip.classList.contains("hidden")) return;
    moveTooltip(evt.clientX, evt.clientY);
  });
  content.addEventListener("mouseout", (evt) => {
    if (!activeChip) return;
    const related = evt.relatedTarget;
    if (related && activeChip.contains && activeChip.contains(related)) return;
    activeChip = null;
    hideTooltip();
  });
});
