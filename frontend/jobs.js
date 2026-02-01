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

function fmtPct(p) {
  const v = Math.round((Number(p) || 0) * 100);
  return `${v}%`;
}

window.addEventListener("DOMContentLoaded", async () => {
  const token = accessToken();
  const content = document.getElementById("content");
  if (!token) {
    content.textContent = "Missing token. Build a SkillPack first in this browser (or open via a share link).";
    return;
  }

  let data;
  try {
    data = await fetchJson("/api/jobs?limit=50", token);
  } catch (e) {
    content.textContent = `Failed to load jobs: ${e}`;
    return;
  }

  const jobs = data.jobs || [];
  if (jobs.length === 0) {
    content.innerHTML = `<div class="subtle">No jobs yet.</div>`;
    return;
  }

  const tokenQ = token ? `?token=${encodeURIComponent(token)}` : "";
  content.innerHTML = `
    <div class="subtle">Showing up to ${jobs.length} recent jobs.</div>
    <div class="section">
      ${jobs
        .map((j) => {
          const jobId = j.job_id;
          const status = j.status;
          const stage = j.stage;
          const err = j.error ? `<div class="excerpt">Error: ${escapeHtml(j.error)}</div>` : "";
          const packId = j.result && j.result.pack_id ? String(j.result.pack_id) : "";
          const packLink = packId
            ? `<a class="link" href="/pack/${escapeHtml(packId)}${tokenQ}">Open pack</a>`
            : `<span class="subtle">No pack yet</span>`;
          const eventsLink = `<a class="link" href="/job/${escapeHtml(jobId)}/events${tokenQ}">Events</a>`;
          const jobLink = `<a class="link" href="/job/${escapeHtml(jobId)}${tokenQ}">${escapeHtml(jobId)}</a>`;
          const metrics = j.metrics ? `<pre class="pre">${escapeHtml(JSON.stringify(j.metrics, null, 2))}</pre>` : "";
          return `
            <div class="rule">
              <div class="rule-title">${jobLink}</div>
              <div class="subtle">status=${escapeHtml(status)} · stage=${escapeHtml(stage)} · progress=${escapeHtml(
            fmtPct(j.progress)
          )} · attempt=${escapeHtml(String(j.attempt || 0))}</div>
              <div class="subtle">created_at=${escapeHtml(j.created_at)} · updated_at=${escapeHtml(j.updated_at)}</div>
              <div class="row" style="margin-top:8px">
                <div class="grow">${packLink}</div>
                <div class="actions">${eventsLink}</div>
              </div>
              ${err}
              ${metrics}
            </div>
          `;
        })
        .join("")}
    </div>
  `;
});

