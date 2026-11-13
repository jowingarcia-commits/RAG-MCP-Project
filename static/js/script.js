const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send");

const SUGGESTIONS = [
  "Summarize the Titanic dataset",
  "Survival rate by passenger class and sex",
  "Which columns have missing values?",
  "Search the web for the history of the Titanic",
];



function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}
function table(rows) {
  const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const hasSep = rows.length > 1 && /^[\s|:-]+$/.test(rows[1]);
  const head = cells(rows[0]);
  const body = rows.slice(hasSep ? 2 : 1).map(cells);
  const th = head.map((c) => "<th>" + inline(c) + "</th>").join("");
  const tr = body.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("");
  return "<div class=\"tablewrap\"><table><thead><tr>" + th + "</tr></thead><tbody>" + tr + "</tbody></table></div>";
}
function render(text) {
  const lines = text.split("\n");
  const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isItem = (l) => /^\s*[-*] /.test(l);
  let html = "", i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++; html += "<pre>" + esc(code.join("\n")) + "</pre>"; continue;
    }
    if (isRow(line)) {
      const rows = [];
      while (i < lines.length && isRow(lines[i])) rows.push(lines[i++]);
      html += table(rows); continue;
    }
    if (isItem(line)) {
      const items = [];
      while (i < lines.length && isItem(lines[i])) items.push(lines[i++].replace(/^\s*[-*] /, ""));
      html += "<ul>" + items.map((t) => "<li>" + inline(t) + "</li>").join("") + "</ul>"; continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !lines[i].startsWith("```") && !isRow(lines[i]) && !isItem(lines[i])) para.push(lines[i++]);
    html += "<p>" + para.map(inline).join("<br>") + "</p>";
  }
  return html;
}

function addMessage(kind, content, tools) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  if (kind === "assistant") el.innerHTML = render(content);
  else el.textContent = content;
  if (tools && tools.length) {
    const d = document.createElement("details");
    d.className = "trace";
    const names = [...new Set(tools.map((t) => t.tool))].join(", ");
    d.innerHTML = "<summary>Used " + esc(names) + "</summary><pre>" + esc(JSON.stringify(tools, null, 2)) + "</pre>";
    el.appendChild(d);
  }
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

async function ask(text) {
  text = text.trim();
  if (!text) return;
  $("suggestions").hidden = true;
  addMessage("user", text);
  input.value = ""; input.style.height = "auto";
  send.disabled = true;
  const wait = addMessage("assistant thinking", "Working on it…");
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json().catch(() => ({}));
    wait.remove();
    if (!res.ok) addMessage("assistant error", data.error || "Something went wrong.");
    else addMessage("assistant", data.answer, data.tools);
  } catch (err) {
    wait.remove();
    addMessage("assistant error", "Could not reach the server. Check that the app is running.");
  } finally {
    send.disabled = false; input.focus();
  }
}

async function loadDatasets() {
  let datasets = [];
  try {
    const res = await fetch("/api/datasets");
    if (res.ok) ({ datasets } = await res.json());
  } catch (err) { /* server down: show the empty state */ }
  $("datasets").innerHTML = datasets.length
    ? datasets.map((d) => "<li><span>" + esc(d.name) + "</span><span class=\"size\">" + d.size_kb + " KB</span></li>").join("")
    : "<li>No datasets yet</li>";
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
});
input.addEventListener("input", () => {
  input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

$("file").addEventListener("change", async (e) => {
  const status = $("upload-status");
  const file = e.target.files[0];
  if (!file) return;
  status.className = ""; status.textContent = "Uploading…";
  const body = new FormData(); body.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { status.className = "error"; status.textContent = data.error || "Upload failed."; }
    else { status.textContent = data.filename + ": " + data.note; loadDatasets(); }
  } catch (err) {
    status.className = "error"; status.textContent = "Could not reach the server.";
  } finally {
    e.target.value = "";
  }
});

$("reset").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  log.innerHTML = ""; $("suggestions").hidden = false;
});

$("suggestions").innerHTML = SUGGESTIONS.map((s) => "<button type=\"button\">" + esc(s) + "</button>").join("");
$("suggestions").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") ask(e.target.textContent); });
loadDatasets();
