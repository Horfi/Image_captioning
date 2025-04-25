// FILE: src/utils/api.js
export async function fetchCaption(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("http://localhost:8000/api/caption", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => null);
    throw new Error(err?.detail || "Server error when generating caption");
  }

  const { caption, processing_time_ms } = await response.json();
  return { caption, processing_time_ms };
}
