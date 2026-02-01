async function uploadOne(file) {
  const ownerToken = localStorage.getItem("paper_skill_owner_token");
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/pdfs/upload", {
    method: "POST",
    headers: ownerToken ? { "X-Owner-Token": ownerToken } : {},
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  if (data.owner_token) localStorage.setItem("paper_skill_owner_token", data.owner_token);
  return data.pdf_id;
}

async function buildPack(payload) {
  const ownerToken = localStorage.getItem("paper_skill_owner_token");
  payload.owner_token = ownerToken;
  const res = await fetch("/api/skillpacks/build", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(ownerToken ? { "X-Owner-Token": ownerToken } : {}),
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  if (data.owner_token) localStorage.setItem("paper_skill_owner_token", data.owner_token);
  return data;
}

function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: ${id}`);
  return el;
}

window.addEventListener("DOMContentLoaded", () => {
  const form = byId("build-form");
  const status = byId("status");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    status.textContent = "Uploading PDFs...";

    const files = byId("pdfs").files;
    if (!files || files.length !== 3) {
      status.textContent = "Please select exactly 3 PDF files.";
      return;
    }

    try {
      const pdfIds = [];
      for (const f of files) {
        pdfIds.push(await uploadOne(f));
      }
      status.textContent = "Starting build job...";

      const payload = {
        pdf_ids: pdfIds,
        pack_name: byId("pack_name").value,
        field_hint: byId("field_hint").value || null,
        target_venue_hint: byId("target_venue_hint").value || null,
        language: "English",
      };
      const { job_id } = await buildPack(payload);
      const ownerToken = localStorage.getItem("paper_skill_owner_token");
      const q = ownerToken ? `?token=${encodeURIComponent(ownerToken)}` : "";
      window.location.href = `/job/${job_id}${q}`;
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    }
  });
});
