function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: ${id}`);
  return el;
}

function jobIdFromPath() {
  const m = window.location.pathname.match(/\/job\/(.+)$/);
  if (!m) throw new Error("Bad job path");
  return m[1];
}

function accessToken() {
  const q = new URLSearchParams(window.location.search);
  const t = q.get("token");
  if (t) return t;
  return localStorage.getItem("paper_skill_owner_token");
}

function render(job) {
  byId("job-meta").textContent = `Status: ${job.status} | Stage: ${job.stage}`;
  const pct = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
  byId("bar").style.width = `${pct}%`;
  byId("pct").textContent = `${pct}%`;
  byId("details").textContent = JSON.stringify(job, null, 2);
}

window.addEventListener("DOMContentLoaded", async () => {
  const jobId = jobIdFromPath();
  const token = accessToken();
  if (token && !localStorage.getItem("paper_skill_owner_token")) {
    localStorage.setItem("paper_skill_owner_token", token);
  }
  const eventsLink = document.getElementById("events_link");
  if (eventsLink) {
    const q = token ? `?token=${encodeURIComponent(token)}` : "";
    eventsLink.href = `/job/${jobId}/events${q}`;
  }
  let done = false;

  while (!done) {
    try {
      const token = accessToken();
      const jobRes = await fetch(`/api/jobs/${jobId}`, {
        headers: token ? { "X-Pack-Token": token } : {},
      });
      if (!jobRes.ok) throw new Error(await jobRes.text());
      const job = await jobRes.json();
      render(job);
      if (job.status === "succeeded" && job.result && job.result.pack_id) {
        const q = token ? `?token=${encodeURIComponent(token)}` : "";
        window.location.href = `/pack/${job.result.pack_id}${q}`;
        return;
      }
      if (job.status === "failed") {
        done = true;
        return;
      }
    } catch (err) {
      byId("details").textContent = `Error: ${err.message}`;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
});
