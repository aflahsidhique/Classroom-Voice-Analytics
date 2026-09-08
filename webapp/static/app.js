const PALETTE = { Teacher: "#2563eb", Student: "#f59e0b", Silence: "#cbd5e1" };

const resultsSection = document.getElementById("results");
const progressWrap = document.getElementById("progress-wrap");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const errorBox = document.getElementById("error-box");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function renderResult(result) {
  resultsSection.hidden = false;

  document.getElementById("summary-text").textContent = result.summary;

  const metrics = result.metrics;

  const cardsEl = document.getElementById("metric-cards");
  cardsEl.innerHTML = "";
  metrics.metrics.forEach((m) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<div class="label">${m.name}</div><div class="value">${m.value}</div>`;
    cardsEl.appendChild(card);
  });

  const formulasEl = document.getElementById("formulas-body");
  formulasEl.innerHTML = metrics.metrics
    .map(
      (m) => `<div class="formula-row"><strong>${m.name}</strong> — <code>${m.formula}</code>
        <br>${m.explanation}<br><em>Interpretation:</em> ${m.interpretation}</div>`
    )
    .join("");

  const total = metrics.teacher_talk_sec + metrics.student_talk_sec + metrics.silence_sec;
  const barEl = document.getElementById("talktime-bar");
  barEl.innerHTML = "";
  if (total > 0) {
    [
      ["Teacher", metrics.teacher_talk_sec],
      ["Student", metrics.student_talk_sec],
      ["Silence", metrics.silence_sec],
    ].forEach(([label, secs]) => {
      const pct = (100 * secs) / total;
      if (pct <= 0) return;
      const seg = document.createElement("div");
      seg.style.width = pct + "%";
      seg.style.background = PALETTE[label];
      seg.title = `${label}: ${pct.toFixed(1)}%`;
      barEl.appendChild(seg);
    });
  }
  document.getElementById("talktime-legend").innerHTML = Object.entries(PALETTE)
    .map(([label, color]) => `<span><span class="swatch" style="background:${color}"></span>${label}</span>`)
    .join("");

  const highlightsEl = document.getElementById("highlights-list");
  highlightsEl.innerHTML = (result.highlights || []).map((h) => `<li>${escapeHtml(h)}</li>`).join("");

  const tbody = document.querySelector("#transcript-table tbody");
  tbody.innerHTML = "";
  (result.turns || []).forEach((t) => {
    const tr = document.createElement("tr");
    tr.className = `speaker-${t.speaker}`;
    const flags = (t.is_question ? "❓" : "") + (t.is_response ? "✅" : "");
    tr.innerHTML = `<td class="time">${fmtTime(t.start)}</td><td>${t.speaker}</td><td>${flags}</td><td>${escapeHtml(t.text)}</td>`;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --- Sample sessions ---
document.getElementById("load-sample-btn").addEventListener("click", async () => {
  const name = document.getElementById("sample-select").value;
  if (!name) return;
  clearError();
  progressWrap.hidden = true;
  try {
    const res = await fetch(`/api/samples/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error("Failed to load sample");
    renderResult(await res.json());
  } catch (err) {
    showError(err.message);
  }
});

// --- Upload + live streaming ---
document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  resultsSection.hidden = true;

  const fileInput = document.getElementById("audio-file");
  const language = document.getElementById("language").value.trim();
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("audio", fileInput.files[0]);
  if (language) formData.append("language", language);

  progressWrap.hidden = false;
  progressFill.style.width = "0%";
  progressLabel.textContent = "Uploading...";

  let uploadRes;
  try {
    uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
  } catch (err) {
    showError("Upload failed: " + err.message);
    progressWrap.hidden = true;
    return;
  }
  if (!uploadRes.ok) {
    const body = await uploadRes.json().catch(() => ({}));
    showError(body.error || "Upload failed");
    progressWrap.hidden = true;
    return;
  }

  const { job_id } = await uploadRes.json();
  const source = new EventSource(`/api/stream/${job_id}`);

  source.onmessage = (event) => {
    const partial = JSON.parse(event.data);
    const total = partial.total_duration_sec || partial.duration_sec || 1;
    const pct = Math.min(100, (100 * partial.duration_sec) / total);
    progressFill.style.width = pct.toFixed(1) + "%";
    progressLabel.textContent = partial.is_final
      ? "Done."
      : `Transcribed ${partial.duration_sec.toFixed(0)}s / ${total.toFixed(0)}s...`;
    renderResult(partial);
  };

  // Pipeline failures (bad audio, transcription error, ...) - a custom SSE
  // event, distinct from EventSource's built-in "error" so it doesn't
  // collide with connection-level failures below.
  source.addEventListener("pipeline_error", (event) => {
    const message = JSON.parse(event.data).message || "Processing failed.";
    showError(message);
    source.close();
  });

  source.addEventListener("done", () => {
    source.close();
  });

  // Native EventSource error: the connection itself dropped (server
  // restarted, network blip, ...), not a pipeline failure.
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) return; // already closed cleanly above
    showError("Lost connection to the server while processing.");
    source.close();
  };
});
