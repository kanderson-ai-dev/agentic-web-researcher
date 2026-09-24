/* Operator console: login, submit, live SSE progress (via fetch so the
   Authorization header can be sent — EventSource cannot), HITL review and
   final report rendering. Plain JS, no build step. */

const $ = (id) => document.getElementById(id);
const show = (id) => $(id).classList.remove("hidden");
const hide = (id) => $(id).classList.add("hidden");

let token = sessionStorage.getItem("awr_token") || null;
let currentJob = null;
let editing = false;

function authHeaders() {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) },
  });
  return resp;
}

function setAuthUI(signedIn) {
  $("auth-status").textContent = signedIn ? "signed in" : "signed out";
  if (signedIn) {
    hide("login-panel");
    show("research-panel");
  } else {
    show("login-panel");
    hide("research-panel");
  }
}

async function trySession() {
  // Auth may be disabled server-side — probe with a cheap GET.
  const resp = await api("/api/v1/research");
  if (resp.status === 200) { setAuthUI(true); return; }
  setAuthUI(Boolean(token));
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("login-error").textContent = "";
  const resp = await fetch("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: $("login-username").value,
      password: $("login-password").value,
    }),
  });
  if (resp.status === 200) {
    token = (await resp.json()).access_token;
    sessionStorage.setItem("awr_token", token);
    setAuthUI(true);
  } else if (resp.status === 503) {
    // Auth disabled server-side: no token needed.
    token = null;
    setAuthUI(true);
  } else {
    $("login-error").textContent = "Sign-in failed.";
  }
});

/* Internal graph node names must never reach the UI: map each one to a
   human-readable phase label, and collapse consecutive repeats of the same
   phase into a single line with a counter. */
const NODE_LABELS = {
  __job__: "Job",
  input_guardrail: "Screening topic",
  planner: "Planning research questions",
  search_worker: "Searching the web",
  aggregate_search: "Collecting search results",
  scrape_worker: "Fetching pages",
  aggregate_documents: "Collecting documents",
  document_worker: "Parsing & sanitizing content",
  to_critic: "Preparing evidence",
  critic: "Evaluating coverage",
  replan: "Planning follow-up research",
  writer: "Writing report",
  output_guardrail: "Verifying citations",
  human_review: "Submitting for review",
  report_assembler: "Assembling final report",
  rejection_output: "Request rejected",
};

const STATUS_LABELS = {
  started: "started",
  completed: "done",
  failed: "failed",
  awaiting_review: "awaiting your review",
};

function logEvent(ev) {
  const label = NODE_LABELS[ev.node] || ev.node;
  const status = STATUS_LABELS[ev.status] || ev.status;
  const time = new Date(ev.ts).toLocaleTimeString();
  const log = $("event-log");
  const last = log.lastElementChild;

  // Collapse consecutive events from the same phase into "label ×N".
  if (last && last.dataset.node === ev.node && ev.node !== "__job__") {
    const count = Number(last.dataset.count || 1) + 1;
    last.dataset.count = String(count);
    last.innerHTML =
      `<span class="node">${label}</span> ${status}` +
      `<span class="count">×${count}</span>` +
      `<span class="time">${time}</span>`;
    return;
  }
  const li = document.createElement("li");
  li.dataset.node = ev.node;
  li.innerHTML = `<span class="node">${label}</span> ${status}` +
    `<span class="time">${time}</span>`;
  log.appendChild(li);
}

$("topic-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const resp = await api("/api/v1/research", {
    method: "POST",
    body: JSON.stringify({ topic: $("topic-input").value, depth: $("depth-select").value }),
  });
  if (!resp.ok) {
    alert(`Submit failed (${resp.status})`);
    return;
  }
  currentJob = await resp.json();
  $("event-log").innerHTML = "";
  $("job-status").textContent = currentJob.status;
  show("progress-panel");
  hide("review-panel");
  hide("result-panel");
  streamJob(currentJob.id);
});

async function streamJob(jobId) {
  const resp = await fetch(`/api/v1/research/${jobId}/stream`, { headers: authHeaders() });
  if (!resp.ok || !resp.body) return;
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      const ev = JSON.parse(dataLine.slice(6));
      handleEvent(ev);
    }
  }
}

function handleEvent(ev) {
  logEvent(ev);
  if (ev.node !== "__job__") return;
  if (ev.status === "awaiting_review") {
    $("job-status").textContent = "awaiting_review";
    openReview(JSON.parse(ev.detail || "{}"));
  } else if (ev.status === "completed" || ev.status === "failed") {
    $("job-status").textContent = ev.status;
    loadResult(currentJob.id);
  }
}

function openReview(payload) {
  $("review-report").textContent = payload.report || "(no report)";
  $("review-meta").textContent = payload.escalated
    ? "Escalated: the critic could not resolve all coverage gaps."
    : "Routine review before publishing.";
  $("edit-area").value = payload.report || "";
  hide("edit-area");
  editing = false;
  show("review-panel");
}

$("approve-btn").addEventListener("click", () => submitReview({ action: "approve" }));
$("reject-btn").addEventListener("click", () => submitReview({ action: "reject" }));
$("edit-btn").addEventListener("click", () => {
  if (!editing) {
    editing = true;
    show("edit-area");
    $("edit-btn").textContent = "Submit edited report";
  } else {
    submitReview({ action: "edit", report: $("edit-area").value });
  }
});

async function submitReview(decision) {
  const resp = await api(`/api/v1/research/${currentJob.id}/review`, {
    method: "POST",
    body: JSON.stringify(decision),
  });
  if (resp.status === 202) {
    hide("review-panel");
    $("job-status").textContent = "running";
  } else {
    alert(`Review failed (${resp.status})`);
  }
}

function renderMarkdown(md) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc(md)
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/\n{2,}/g, "<br/><br/>");
}

async function loadResult(jobId) {
  const resp = await api(`/api/v1/research/${jobId}`);
  if (!resp.ok) return;
  const job = await resp.json();
  show("result-panel");
  $("report-body").innerHTML = job.report
    ? renderMarkdown(job.report)
    : "<p><em>No report produced.</em></p>";
  $("source-list").innerHTML = "";
  for (const s of job.sources || []) {
    const li = document.createElement("li");
    li.innerHTML = `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${s.title || s.url}</a>`;
    $("source-list").appendChild(li);
  }
  $("citation-list").innerHTML = "";
  for (const c of job.citations || []) {
    const li = document.createElement("li");
    li.innerHTML = `${c.claim}<blockquote>${c.quote}</blockquote>`;
    $("citation-list").appendChild(li);
  }
  $("job-meta").textContent =
    `job ${job.id} · status ${job.status} · est. cost $${(job.cost_usd || 0).toFixed(6)}` +
    ` · ${(job.timings?.total_seconds ?? 0)}s`;
}

trySession();
