function jobIdFromPath() {
  const m = window.location.pathname.match(/\/job\/(.+)\/events$/);
  if (!m) throw new Error("Bad job events path");
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

window.addEventListener("DOMContentLoaded", async () => {
  const jobId = jobIdFromPath();
  const token = accessToken();
  document.getElementById("meta").textContent = `job_id=${jobId}`;
  if (!token) {
    document.getElementById("events").textContent = "Missing token.";
    return;
  }

  const job = await fetchJson(`/api/jobs/${jobId}`, token);
  const events = await fetchJson(`/api/jobs/${jobId}/events`, token);

  document.getElementById("metrics").textContent = JSON.stringify(job.metrics || {}, null, 2);

  const rows = (events.events || []).map((ev) => {
    const data = ev.data && typeof ev.data === "object" ? ev.data : null;
    const tb = data && data.traceback ? String(data.traceback) : null;
    const dataNoTb = data ? { ...data } : null;
    if (dataNoTb && "traceback" in dataNoTb) delete dataNoTb.traceback;

    const dataHtml =
      dataNoTb && Object.keys(dataNoTb).length > 0
        ? `<div class="subtle">data</div><pre class="pre">${escapeHtml(JSON.stringify(dataNoTb, null, 2))}</pre>`
        : "";
    const tbHtml = tb
      ? `<details><summary class="evidence">traceback</summary><pre class="pre">${escapeHtml(tb)}</pre></details>`
      : "";

    return `<div class="rule">
      <div class="rule-title">${escapeHtml(ev.level)} · ${escapeHtml(ev.stage)}</div>
      <div class="subtle">${escapeHtml(ev.ts)}</div>
      <div class="rule-body">${escapeHtml(ev.message || "")}</div>
      ${dataHtml}
      ${tbHtml}
    </div>`;
  });

  const container = document.getElementById("events");
  if (rows.length === 0) {
    container.innerHTML = `<div class="subtle">No events.</div>`;
  } else {
    container.innerHTML = rows.join("");
  }
});
