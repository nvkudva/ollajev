const $ = (s, r = document) => r.querySelector(s);
const store = {
  get(k, fb) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fb; } catch { return fb; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ---- markup --------------------------------------------------------------
// Components are functions from state to an HTML string. `html` returns a `Raw`, so a component
// can be interpolated into another without being escaped twice; everything else is escaped, which
// is what keeps user text — the state, instructions, question names — from becoming markup.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
class Raw { constructor(s) { this.s = s; } }

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

function slot(v) {
  if (v == null || v === false) return "";
  if (v instanceof Raw) return v.s;
  if (Array.isArray(v)) return v.map(slot).join("");
  return esc(v);
}

const html = (strings, ...values) =>
  new Raw(strings.reduce((out, s, i) => out + s + (i < values.length ? slot(values[i]) : ""), ""));

/** Markup → one detached element, for the places that append rather than replace. */
function node(markup) {
  const t = document.createElement("template");
  t.innerHTML = markup.s;
  return t.content.firstElementChild;
}

/** Replace a container's markup, then put the caret back where it was. Adding or removing a row
    rebuilds the list, and without this the field being edited loses focus and selection. */
function mount(host, markup) {
  const a = document.activeElement;
  const key = a?.dataset?.key;
  const caret = a?.selectionStart ?? null;
  host.innerHTML = markup.s;
  if (!key) return;
  const next = host.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (!next) return;
  next.focus();
  if (caret != null && next.setSelectionRange) next.setSelectionRange(caret, caret);
}

// ---- theme ---------------------------------------------------------------

let theme = store.get("ollajev.theme", "system");
function applyTheme() {
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("#theme").textContent = theme;
  $("#theme").setAttribute("aria-label", `Colour theme: ${theme}. Click to change.`);
}
$("#theme").onclick = () => {
  theme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  store.set("ollajev.theme", theme);
  applyTheme();
};
applyTheme();

// ---- question set editor -------------------------------------------------

const DEFAULT_SET = [
  { name: "intent", type: "choice", instructions: "What does the customer want?",
    criteria: [["refund", "money returned or a duplicate charge reversed"],
               ["technical_help", "a bug, outage or integration problem"],
               ["other", "none of the other options fits"]] },
  { name: "urgency", type: "score", instructions: "How urgent is this message?",
    criteria: ["Can wait", "Needs attention this week", "Needs attention today"] },
  { name: "is_frustrated", type: "noul", instructions: "Does the customer sound frustrated?", criteria: [] },
];

let questions = store.get("ollajev.questions", DEFAULT_SET);
const save = () => store.set("ollajev.questions", questions);

const modelSel = $("#model");
let limits = {};
let models = [];   // downloaded, from /v1/models
let catalog = [];  // curated models not downloaded yet, from /ui/catalog
function downloaded(name) { return models.some((m) => m.name === name); }
const currentModel = () => modelSel.value || "jev-latest";
const stateBox = $("#state");
stateBox.value = store.get("ollajev.state", "");

const qJsonHost = $("#q-json");
const requestError = $("#request-error");

/* Attached media, kept in memory only: data URLs are too big for localStorage. */
let attachments = [];
const MEDIA_FIELD = { image: "images", audio: "audio", video: "videos" };

/** `images`, `audio` and `videos` as data URLs. `short` cuts each to its type and file name, for the JSON pane and
    the saved conversation. */
function mediaFields({ short = false } = {}) {
  const out = {};
  for (const a of attachments) {
    (out[a.field] ??= []).push(short ? `${a.url.slice(0, a.url.indexOf(",") + 1)}… (${a.name})` : a.url);
  }
  return out;
}

/** The wire request the sidebar holds, in the order the server documents it. */
function requestBody({ strict = true, short = false } = {}) {
  return { state: stateBox.value, model: currentModel(), ...mediaFields({ short }), questions: buildQuestions({ strict }) };
}

function failRequest(message) {
  requestError.textContent = message;
  requestError.hidden = false;
  return false;
}

/* The JSON view is a plain textarea: the page's CSP allows scripts only from this server, and it
   works offline. */
const jsonPane = {
  read() { return qJsonHost.value; },
  write(text) { qJsonHost.value = text; },
};
qJsonHost.addEventListener("input", () => { requestError.hidden = true; });

function blankFor(type) {
  return {
    name: "", type, instructions: "",
    criteria: type === "choice" ? [["", ""]] : type === "score" ? ["", "", ""] : [],
  };
}

const REMOVE_LABEL = { "del-question": "Remove this question", "del-option": "Remove this option", "del-rung": "Remove this rung" };

/** One question card. Every input carries a `data-field` path into `questions`, which is how the
    delegated handler below writes back without a closure per node. */
function questionMarkup(q, i) {
  // Raw: these are attributes spliced mid-tag, so they must not be escaped as text.
  const field = (path) => html`data-field="${i}.${path}" data-key="${i}.${path}"`;
  const remove = (act, j) => html`
    <button class="ghost x" type="button" data-act="${act}" data-i="${i}" data-j="${j}"
            data-key="${act}-${i}-${j}" aria-label="${REMOVE_LABEL[act]}">✕</button>`;

  const parts = [html`
    <div class="row">
      <input name="name" placeholder="question_name" aria-label="Question ${i + 1} name" value="${q.name}" ${field("name")}>
      <span class="meta">${q.type}</span>
      ${remove("del-question", 0)}
    </div>
    <textarea placeholder="instructions" aria-label="Question ${i + 1} instructions" ${field("instructions")}>${q.instructions}</textarea>`];

  if (q.type === "choice") {
    parts.push(html`<span class="q-label">criteria — label : description</span>`);
    q.criteria.forEach((pair, j) => parts.push(html`
      <div class="crit">
        <input class="label" placeholder="label" aria-label="Option ${j + 1} label" value="${pair[0]}" ${field(`criteria.${j}.0`)}>
        <input placeholder="description (optional)" aria-label="Option ${j + 1} description" value="${pair[1] ?? ""}" ${field(`criteria.${j}.1`)}>
        ${remove("del-option", j)}
      </div>`));
    parts.push(html`<button class="ghost" type="button" data-act="add-option" data-i="${i}" data-key="add-option-${i}">+ option</button>`);
  } else if (q.type === "score") {
    parts.push(html`<span class="q-label">criteria — lowest rung first</span>`);
    q.criteria.forEach((rung, j) => parts.push(html`
      <div class="crit">
        <input placeholder="rung ${j}" aria-label="Rung ${j}" value="${rung}" ${field(`criteria.${j}`)}>
        ${remove("del-rung", j)}
      </div>`));
    parts.push(html`<button class="ghost" type="button" data-act="add-rung" data-i="${i}" data-key="add-rung-${i}">+ rung</button>`);
  } else {
    parts.push(html`<span class="q-label">criteria (optional)</span>`);
    ["true", "false"].forEach((label, j) => parts.push(html`
      <div class="crit">
        <input class="label" value="${label}" aria-label="Answer ${label}" disabled>
        <input placeholder="what makes it ${label}" aria-label="What makes it ${label}" value="${q.criteria[j] ?? ""}" ${field(`criteria.${j}`)}>
      </div>`));
  }

  return html`<div class="q">${parts}</div>`;
}

/** Editor rows → the Jev `questions` object. Throws on user error; with `strict: false` it
 *  emits whatever is on screen, so the JSON view can always render the current rows. */
function buildQuestions({ strict = true } = {}) {
  const out = {};
  for (const [i, q] of questions.entries()) {
    const name = q.name.trim();
    const touched = q.instructions.trim() || q.criteria.flat().some((c) => (c ?? "").trim());
    if (!name && !touched) continue;
    if (!name && strict) throw new Error(`Question ${i + 1} (${q.type}) needs a name.`);
    if (name in out) {
      if (strict) throw new Error(`Duplicate question name '${name}'.`);
      continue;
    }
    const body = { type: q.type };
    if (q.instructions.trim()) body.instructions = q.instructions.trim();
    if (q.type === "choice") {
      const criteria = {};
      for (const [label, desc] of q.criteria) {
        const l = label.trim();
        if (!l) continue;
        criteria[l] = desc.trim() || null;
      }
      if (Object.keys(criteria).length < 2 && strict) throw new Error(`'${name}' needs at least two labelled options.`);
      body.criteria = criteria;
    } else if (q.type === "score") {
      const rungs = q.criteria.map((r) => r.trim()).filter(Boolean);
      if (!rungs.length && strict) throw new Error(`'${name}' needs at least one rung.`);
      body.criteria = rungs;
    } else {
      const [t, f] = q.criteria.map((c) => (c ?? "").trim());
      if (t || f) body.criteria = { true: t || null, false: f || null };
    }
    out[name] = body;
  }
  if (!Object.keys(out).length && strict) throw new Error("Add at least one question.");
  return out;
}

/** Answer to a `data-field` path — "0.criteria.1.0" — written back into `questions`. */
function setField(path, value) {
  const parts = path.split(".");
  const leaf = parts.pop();
  let at = questions;
  for (const p of parts) at = at[p];
  at[leaf] = value;
  save();
}

/** Load a named example: its sample state into the textarea, its questions into the editor, and its sample files
    (served from /static) in place of whatever was attached. */
async function adoptExample(example) {
  attachments.length = 0;
  for (const src of example.images ?? []) {
    try {
      const blob = await (await fetch(src)).blob();
      attachments.push({ kind: "image", field: "images", name: src.split("/").pop(), url: await readDataUrl(blob) });
    } catch {
      failRequest(`The example image ${src} could not be loaded; attach your own image instead.`);
    }
  }
  paintMedia();
  stateBox.value = example.state;
  store.set("ollajev.state", example.state);
  syncSend();
  // adoptPreset re-renders, which refreshes the JSON pane from the new state when it is showing.
  adoptPreset(example.questions);
}

/** Server preset (already Jev-shaped) → editor rows. */
function adoptPreset(set) {
  questions = Object.entries(set).map(([name, q]) => ({
    name, type: q.type, instructions: q.instructions ?? "",
    criteria: q.type === "choice" ? Object.entries(q.criteria).map(([l, d]) => [l, d ?? ""])
      : q.type === "score" ? q.criteria.slice()
      : [q.criteria?.true ?? "", q.criteria?.false ?? ""],
  }));
  store.set("ollajev.questions", questions);
  paintEditor();
}

let qView = "ui";

/** JSON view text → the sidebar's fields. Shows the error and returns false when it isn't a request. */
function applyJsonView() {
  try {
    const body = JSON.parse(jsonPane.read());
    if (typeof body.state !== "string") {
      throw new Error("state must be a string to edit in UI mode — switch back to JSON to use an object or array.");
    }
    if (!body.questions || typeof body.questions !== "object") {
      throw new Error("the request needs a `questions` object.");
    }
    stateBox.value = body.state;
    store.set("ollajev.state", body.state);
    syncSend();
    adoptPreset(body.questions);
    return true;
  } catch (e) {
    requestError.textContent = `Not a valid request: ${e.message}`;
    requestError.hidden = false;
    return false;
  }
}

/** Send what the sidebar holds — the JSON pane first, if that is what is on screen. */
function sendRequest() {
  if (busy) return;
  if (qView === "json" && !applyJsonView()) return;
  let request;
  try {
    request = requestBody();
  } catch (e) {
    return failRequest(e.message);
  }
  if (!request.state.trim() && !attachments.length) return failRequest("The request needs a state or media to judge.");
  requestError.hidden = true;
  ask(request);
}

/** The state is the one thing a request cannot go without, so Send stands down while it is blank. */
let busy = false;
function syncSend() {
  $("#send").disabled = busy || (!stateBox.value.trim() && !attachments.length) || (modelSel.value !== "" && !downloaded(modelSel.value));
}

function setView(mode) {
  if (mode === qView) return;
  if (qView === "json" && !applyJsonView()) return;
  requestError.hidden = true;
  qView = mode;
  paintEditor();
}

function paintEditor() {
  mount($("#questions"), html`${questions.map(questionMarkup)}`);
  $("#q-count").textContent = `${questions.length} question${questions.length === 1 ? "" : "s"}`;

  const asJson = qView === "json";
  $("#request-ui").hidden = asJson;
  qJsonHost.hidden = !asJson;
  if (asJson) jsonPane.write(JSON.stringify(requestBody({ strict: false, short: true }), null, 2));
  for (const b of document.querySelectorAll("#q-view button")) {
    b.classList.toggle("is-on", b.dataset.view === qView);
    b.setAttribute("aria-pressed", String(b.dataset.view === qView));
  }
}

for (const b of document.querySelectorAll("#q-view button")) b.onclick = () => setView(b.dataset.view);

// ---- rendering answers ---------------------------------------------------
// One question's answer is one distribution: a single fill on a neutral track,
// no legend, rank carried by ink weight. Same readout for all three types.

const pct = (p) => (p * 100 >= 9.95 ? (p * 100).toFixed(0) : (p * 100).toFixed(1));

function rowsFor(a) {
  if (a.type === "noul") {
    const p = a.noul ?? 0;
    return [{ label: "false", p: 1 - p }, { label: "true", p }];
  }
  const probs = Object.entries(a.probabilities ?? {});
  if (a.type === "score") {
    const legend = a.legend ?? {};
    return probs.map(([k, p]) => ({ label: `${k} ${legend[k] ?? ""}`.trim(), p }));
  }
  return probs.map(([label, p]) => ({ label, p }));
}

function summaryFor(a) {
  if (a.type === "noul") return (a.noul ?? 0) >= 0.5 ? "true" : "false";
  if (a.type === "score") return (a.score ?? 0).toFixed(2);
  return a.choice ?? "";
}

function answerMarkup(name, q, a) {
  const data = rowsFor(a);
  const winner = data.reduce((best, r, i) => (r.p > data[best].p ? i : best), 0);
  const act = a.action?.act_probability ?? a.rl_agent?.act_probability;
  const foot = a.type === "score" ? `expectation, ${data.length} levels`
    : a.type === "noul" ? "p(true)" : `${data.length} options`;

  return html`
    <section class="dist">
      <div class="dist-q">
        <span class="dist-instr">${q?.instructions || name}</span>
        <span class="dist-answer">${summaryFor(a)}</span>
      </div>
      <header class="dist-head">
        <h3>${name}</h3>
        <span class="dist-type">${a.type}</span>
      </header>
      <ol class="bars">
        ${data.map((r, i) => html`
        <li class="${i === winner ? "bar-row is-top" : "bar-row"}" title="${r.label} — ${r.p.toFixed(4)}">
          <span class="bar-label">${r.label}</span>
          <span class="bar-track">
            <span class="bar-fill" style="width: max(2px, ${(Number(r.p) * 100).toFixed(4)}%)"></span>
          </span>
          <span class="bar-value">${pct(r.p)}<span class="pct">%</span></span>
        </li>`)}
      </ol>
      <footer class="dist-foot">
        <span>${foot}</span>
        <span class="dist-meta">
          ${a.confidence != null ? html`<span>confidence ${a.confidence.toFixed(3)}</span>` : ""}
          ${act != null ? html`<span>act ${act.toFixed(3)}</span>` : ""}
        </span>
      </footer>
    </section>`;
}

function answersMarkup(sent, data) {
  return html`
    <div class="answers">
      ${Object.entries(data.answers ?? {}).map(([name, a]) => answerMarkup(name, sent[name], a))}
    </div>`;
}

function errorMarkup(detail) {
  const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
  const list = Array.isArray(detail) ? detail : [{ loc: [], msg }];
  return html`
    <div class="answers">
      <div class="notice">
        <h3>Request rejected</h3>
        ${list.map((d) => html`<div><code>${(d.loc ?? []).join(".")}</code> ${d.msg}</div>`)}
      </div>
    </div>`;
}

// ---- conversation --------------------------------------------------------

const log = $("#log");
/* Turns saved before the sidebar became a whole request carry `state`/`sent` instead of
   `request`; re-key them so an existing log still renders. */
let history = store.get("ollajev.history", []).map((turn) => turn.request ? turn : {
  request: { state: turn.state ?? "", model: currentModel(), questions: turn.sent ?? {} },
  data: turn.data,
  error: turn.error,
});

/* Index of the one expanded turn, 0-based over `history`. -1 means all of them are minimized. */
let openIndex = -1;

/* Collapse and the UI/JSON switch stay imperative: they only add or lift classes and the `hidden`
   property, so they never rebuild a turn — and never disturb a selection in the log. */
function syncOpen() {
  document.querySelectorAll("#log .turn").forEach((el, i) => {
    const open = i === openIndex;
    el.classList.toggle("is-open", open);
    const caret = el.querySelector(".turn-caret");
    if (caret) caret.textContent = open ? "▾" : "▸";
    el.querySelector(".turn-toggle")?.setAttribute("aria-expanded", String(open));
  });
}

/* JSON is the wire pair — the request as sent and the reply as received — so the answers block,
   which is only a rendering of `response.answers`, stands down while it is up. */
function applyView(turnEl, view) {
  const json = view === "json";
  turnEl.querySelector(".state-text").hidden = json;
  for (const el of turnEl.querySelectorAll(".state-json, .state-sub, .state-resp")) el.hidden = !json;
  const answers = turnEl.querySelector(".answers");
  if (answers) answers.hidden = json;
  for (const b of turnEl.querySelectorAll(".turn-view button[data-view]")) {
    b.classList.toggle("is-on", b.dataset.view === view);
    b.setAttribute("aria-pressed", String(b.dataset.view === view));
  }
}

function turnHeadMarkup(turn, index) {
  const tokens = turn.data?.usage?.input_tokens;
  const maxTokens = models.find((m) => m.name === turn.data?.model)?.limits?.max_tokens;
  const time = turn.at ? new Date(turn.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null;
  const model = turn.data?.model ?? turn.request?.model;
  return html`
    <header class="turn-head" data-act="toggle-turn" data-i="${index}">
      <button type="button" class="turn-toggle" aria-expanded="false"><span class="turn-caret" aria-hidden="true">▸</span> Request #${index + 1}</button>
      ${time ? html`<span>· ${time}</span>` : ""}
      ${model ? html`<span class="turn-model" title="${model}">· ${model}</span>` : ""}
      <span class="turn-meta">
        ${maxTokens && tokens >= 0.9 * maxTokens ? html`<span class="turn-warn" title="Requests past the model's ${maxTokens}-token limit are refused or truncated">⚠ near ${maxTokens}-tok limit</span>` : ""}
        <span>${turn.data ? (tokens != null ? `${tokens} tok` : "done") : turn.pending ? "sending…" : "failed"}</span>
        ${turn.loadMs != null ? html`<span title="Loading the model into memory, before this request could run">load ${turn.loadMs} ms</span>` : ""}
        ${turn.ms != null ? html`<span title="Time the model took to answer">${turn.ms} ms</span>` : ""}
        <div class="seg turn-view" role="group" aria-label="View for request ${index + 1}">
          <button type="button" class="ghost" data-act="turn-view" data-i="${index}" data-view="ui" aria-pressed="true">UI</button>
          <button type="button" class="ghost" data-act="turn-view" data-i="${index}" data-view="json" aria-pressed="false">JSON</button>
        </div>
      </span>
    </header>`;
}

/* Both views are in the markup and swapped with `hidden`, so flipping the toggle never rebuilds. */
function stateRowMarkup(turn) {
  const replied = turn.data !== undefined || turn.error !== undefined;
  return html`
    <div class="state">
      <div class="state-text">${turn.request.state}${attachedNames(turn.request).map((n) => html`<div class="meta">📎 ${n}</div>`)}</div>
      <pre class="state-json">${JSON.stringify(turn.request, null, 2)}</pre>
      ${replied ? html`<div class="state-label state-sub">Response</div>
      <pre class="state-resp">${JSON.stringify(turn.data ?? { error: turn.error }, null, 2)}</pre>` : ""}
    </div>`;
}

/** File names of a shortened request's media: each entry ends in "(name)". */
function attachedNames(request) {
  return Object.values(MEDIA_FIELD).flatMap((f) => request[f] ?? []).map((u) => u.match(/\((.*)\)$/)?.[1] ?? "media");
}

function turnMarkup(turn, index) {
  const body = turn.error ? errorMarkup(turn.error)
    : turn.data ? answersMarkup(turn.request.questions, turn.data)
    : html`<div class="turn-pending">${turn.loading ? `loading ${turn.request.model} into memory…` : "thinking…"}<div class="progress" role="progressbar" aria-label="${turn.loading ? "Loading model" : "Running"}"></div></div>`;
  return html`<div class="turn">${turnHeadMarkup(turn, index)}${stateRowMarkup(turn)}${body}</div>`;
}

function showReadout(turn) {
  if (!turn.data) return;
  const tokens = turn.data.usage?.input_tokens;
  const parts = [];
  if (tokens != null) parts.push(`${tokens} tok`);
  if (turn.ms != null) parts.push(`${turn.ms} ms`);
  if (turn.loadMs != null) parts.push(`+ load ${turn.loadMs} ms`);
  $("#r-tokens").textContent = parts.join(" · ") || "—";
}

const scrollLog = () => { log.parentElement.scrollTop = log.parentElement.scrollHeight; };

function renderLog() {
  if (!history.length) {
    log.innerHTML = html`
      <div class="empty">Build a request on the left and press Send. The model answers every question in
      one pass — each turn is independent, it has no memory.</div>`.s;
    return;
  }
  openIndex = history.length - 1;
  log.innerHTML = html`${history.map(turnMarkup)}`.s;
  /* Both views ship in the markup, so a restored turn has to be told which one it shows. */
  log.querySelectorAll(".turn").forEach((el, i) => applyView(el, history[i].view ?? "ui"));
  syncOpen();
}

/** "load;dur=2710, run;dur=87" -> { load: 2710, run: 87 }. */
function serverTiming(header) {
  const timing = {};
  for (const entry of (header ?? "").split(",")) {
    const [name, ...params] = entry.trim().split(";");
    const duration = params.find((param) => param.trim().startsWith("dur="));
    if (name && duration) timing[name] = Math.round(Number(duration.split("=")[1]));
  }
  return timing;
}

async function ask(request) {
  busy = true;
  syncSend();
  log.querySelector(".empty")?.remove();
  const at = Date.now();
  const loaded = await fetch("/api/ps").then((r) => r.json()).then((b) => b.models.some((m) => m.name === request.model)).catch(() => true);
  const sent = request;
  request = { ...request, ...mediaFields({ short: true }) };
  const pending = node(turnMarkup({ request, at, pending: true, loading: !loaded }, history.length));
  log.append(pending);
  applyView(pending, "ui");
  scrollLog();
  /* The new turn opens; every turn before it minimizes. */
  openIndex = history.length;
  syncOpen();

  let turn;
  const started = performance.now();
  try {
    const res = await fetch("/v1/systemone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(sent),
    });
    const body = await res.json();
    const ms = Math.round(performance.now() - started);
    // The server splits its time into the model load (only when this request loaded it) and the answer.
    const timing = serverTiming(res.headers.get("server-timing"));
    turn = res.ok
      ? { request, data: body, at, ms: timing.run ?? ms, loadMs: timing.load }
      : { request, error: body.detail ?? `HTTP ${res.status}`, at, ms };
  } catch (e) {
    turn = { request, error: `Could not reach the server or read its reply (${e.message ?? e}).`, at };
  } finally {
    busy = false;
    syncSend();
  }
  pending.remove();
  history.push(turn);
  store.set("ollajev.history", history.slice(-30));
  openIndex = history.length - 1;
  const appended = node(turnMarkup(turn, openIndex));
  log.append(appended);
  applyView(appended, "ui");
  syncOpen();
  showReadout(turn);
}

// ---- actions -------------------------------------------------------------
// One listener for generated markup. A node says what it is with `data-act`; the handler reads
// its `data-i` / `data-j` / `data-view` rather than closing over a position in the array.

const ACTIONS = {
  "toggle-turn": (el) => { openIndex = openIndex === Number(el.dataset.i) ? -1 : Number(el.dataset.i); syncOpen(); },
  "turn-view": (el) => {
    const turn = history[Number(el.dataset.i)];
    turn.view = el.dataset.view;
    applyView(el.closest(".turn"), turn.view);
  },
  "del-media": (el) => { attachments.splice(Number(el.dataset.i), 1); paintMedia(); if (qView === "json") paintEditor(); },
  "del-question": (el) => { questions.splice(Number(el.dataset.i), 1); save(); paintEditor(); },
  "add-option": (el) => { questions[Number(el.dataset.i)].criteria.push(["", ""]); save(); paintEditor(); },
  "del-option": (el) => { questions[Number(el.dataset.i)].criteria.splice(Number(el.dataset.j), 1); save(); paintEditor(); },
  "add-rung": (el) => { questions[Number(el.dataset.i)].criteria.push(""); save(); paintEditor(); },
  "del-rung": (el) => { questions[Number(el.dataset.i)].criteria.splice(Number(el.dataset.j), 1); save(); paintEditor(); },
};

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-act]");
  if (el) ACTIONS[el.dataset.act]?.(el);
});

document.addEventListener("input", (e) => {
  const el = e.target.closest("[data-field]");
  if (el) { setField(el.dataset.field, el.value); requestError.hidden = true; }
});

// ---- wiring --------------------------------------------------------------

const addQuestion = (type) => {
  if (qView === "json" && !applyJsonView()) return;
  questions.push(blankFor(type));
  paintEditor();
  $(`#questions [data-key="${questions.length - 1}.name"]`)?.focus();
};
$("#add-noul").onclick = () => addQuestion("noul");
$("#add-choice").onclick = () => addQuestion("choice");
$("#add-score").onclick = () => addQuestion("score");
$("#clear-log").onclick = () => { history = []; store.set("ollajev.history", history); renderLog(); };

$("#send").onclick = sendRequest;
stateBox.addEventListener("input", () => { store.set("ollajev.state", stateBox.value); requestError.hidden = true; syncSend(); });
stateBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sendRequest();
});

paintEditor();
renderLog();
syncSend();

function limitsText(l) {
  const parts = [];
  if (l.max_options) parts.push(`${l.max_options} options`);
  if (l.max_levels) parts.push(`${l.max_levels} levels`);
  if (l.max_tokens) parts.push(`${l.max_tokens} tok`);
  if (l.languages) parts.push(l.languages);
  if (l.inputs) parts.push(l.inputs.join(", "));
  return parts.join(" · ") || "—";
}

// ---- media ---------------------------------------------------------------
// Shown when the model reads images, audio or video, or while something is still attached.

const mediaInput = $("#media");

function paintMedia() {
  const kinds = (limits.inputs ?? []).filter((k) => k in MEDIA_FIELD);
  $("#media-box").hidden = !kinds.length && !attachments.length;
  $("#media-kinds").textContent = kinds.length ? `this model reads ${kinds.join(", ")}` : "this model reads text only";
  mediaInput.accept = kinds.map((k) => `${k}/*`).join(",");
  mount($("#media-list"), html`${attachments.map((a, i) => html`
    <li>
      ${a.kind === "image" ? html`<img src="${a.url}" alt="">` : html`<span class="media-kind">${a.kind}</span>`}
      <span class="media-name" title="${a.name}">${a.name}</span>
      <button class="ghost x" type="button" data-act="del-media" data-i="${i}" data-key="del-media-${i}" aria-label="Remove ${a.name}">✕</button>
    </li>`)}`);
  syncSend();
}

const readDataUrl = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(file);
});

mediaInput.onchange = async () => {
  for (const file of mediaInput.files) {
    const kind = file.type.split("/")[0];
    if (!(kind in MEDIA_FIELD)) {
      failRequest(`${file.name} is not an image, audio or video file.`);
      continue;
    }
    attachments.push({ kind, field: MEDIA_FIELD[kind], name: file.name, url: await readDataUrl(file) });
  }
  mediaInput.value = "";
  paintMedia();
  if (qView === "json") paintEditor();
};

function showModel() {
  const m = models.find((x) => x.name === modelSel.value);
  limits = m?.limits ?? {};
  $("#r-limits").textContent = $("#r-limits").title = m ? limitsText(limits) : "download the model to see its limits";
  showPull();
  paintMedia();
  const repo = modelSel.value.split(":")[0];
  const link = $("#model-link");
  link.href = `https://huggingface.co/${repo}`;
  link.textContent = repo || "the selected model";
}

// The model a request goes to when the page opens: one already in memory, so Send does not load another (the
// saved pick if it is among them), else the server's default, else the last pick on this browser.
function initialModel(names, loaded, defaultName, saved) {
  const inMemory = loaded.filter((name) => names.includes(name));
  if (inMemory.includes(saved)) return saved;
  if (inMemory.length) return inMemory[0];
  if (defaultName) return defaultName;
  return names.includes(saved) ? saved : names[0] ?? "";
}

// The picker lists downloaded models first, then every curated model not on disk yet; picking one of those
// offers to download it here.

async function loadModels(select) {
  try {
    const [body, ps, cat] = await Promise.all([
      fetch("/v1/models").then((r) => r.json()),
      fetch("/api/ps").then((r) => r.json()).catch(() => ({ models: [] })),
      fetch("/ui/catalog").then((r) => r.json()).catch(() => ({ models: [] })),
    ]);
    models = body.models ?? [];
    catalog = (cat.models ?? []).filter((e) => !downloaded(e.name));
    const names = models.map((m) => m.name);
    const loaded = (ps.models ?? []).map((m) => m.name);
    mount(modelSel, html`
      <optgroup label="Downloaded">
        ${models.length ? models.map((m) => html`<option value="${m.name}">${m.name}${m.default ? " (default)" : ""}</option>`)
          : html`<option value="" disabled>No models downloaded yet</option>`}
      </optgroup>
      ${catalog.length ? html`<optgroup label="Available to download">
        ${catalog.map((e) => html`<option value="${e.name}">${e.name} · ${e.size_gb} GB</option>`)}
      </optgroup>` : ""}`);
    const pick = select ?? initialModel(names, loaded, models.find((m) => m.default)?.name, store.get("ollajev.model", ""));
    if (pick) modelSel.value = pick;
    showModel();
  } catch {
    modelSel.append(node(html`<option value="">Could not load models</option>`));
  }
}
loadModels();
modelSel.onchange = () => { store.set("ollajev.model", modelSel.value); showModel(); };

// ---- download ------------------------------------------------------------

let pulling = null;  // the model being downloaded, if any

function showPull() {
  const entry = catalog.find((e) => e.name === modelSel.value);
  $("#pull-row").hidden = !entry && pulling === null;
  if (entry && pulling === null) {
    $("#pull").textContent = `Download ${entry.size_gb} GB`;
    $("#pull-status").textContent = entry.description;
  }
  $("#pull").hidden = !entry || pulling !== null;
  syncSend();
}

const gb = (bytes) => (bytes / 1e9).toFixed(1);

$("#pull").onclick = async () => {
  const name = modelSel.value;
  pulling = name;
  showPull();
  const status = $("#pull-status");
  status.textContent = `starting ${name}…`;
  let error = null;
  try {
    const res = await fetch("/api/pull", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: name, stream: true }),
    });
    if (!res.ok) {  // a 401 or 422 is one JSON body, not the NDJSON stream
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error ?? body.detail?.[0]?.msg ?? `HTTP ${res.status}`);
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (let done = false; !done;) {
      const chunk = await reader.read();
      done = chunk.done;
      buffer += chunk.value ?? "";
      const lines = buffer.split("\n");
      buffer = done ? "" : lines.pop();  // the last line may come without a newline
      for (const line of lines.filter((l) => l.trim())) {
        const event = JSON.parse(line);
        if (event.error) error = event.error;
        else if (event.total) status.textContent = `${name}: ${gb(event.completed)} of ${gb(event.total)} GB`;
        else if (event.status) status.textContent = `${name}: ${event.status}`;
      }
    }
  } catch (e) {
    error = `Download failed: ${e.message ?? e}`;
  }
  pulling = null;
  if (error) {
    showPull();
    status.textContent = error;
    return;
  }
  await loadModels(name);
  status.textContent = "";
};

fetch("/ui/presets").then((r) => r.json()).then((presets) => {
  const sel = $("#preset");
  for (const name of Object.keys(presets)) sel.append(node(html`<option value="${name}">${name}</option>`));
  sel.onchange = () => { if (sel.value) adoptExample(presets[sel.value]); sel.value = ""; };
  // The first visit only opens on the triage example, overwriting whatever the last session left.
  if (!store.get("ollajev.visited")) {
    adoptExample(presets.triage);
    store.set("ollajev.visited", true);
  }
}).catch(() => {});

// ---- curl ----------------------------------------------------------------
// The request on screen (the JSON pane's text when that view is open) as a command to paste in a shell.

/** POSIX single quotes: the text stays exactly as is, and each ' becomes '\''. */
function shellQuote(text) {
  return `'${text.replaceAll("'", `'\\''`)}'`;
}

function curlCommand(body) {
  return [
    `curl -s ${shellQuote(`${location.origin}/v1/systemone`)} \\`,
    `  -H 'content-type: application/json' \\`,
    `  -d ${shellQuote(JSON.stringify(body, null, 2))}`,
  ].join("\n");
}

const curlDialog = $("#curl-dialog");
$("#curl-open").onclick = () => {
  let body;
  try {
    // The JSON pane shows media cut short; the command carries them whole.
    body = qView === "json" ? { ...JSON.parse(jsonPane.read()), ...mediaFields() } : requestBody({ strict: false });
  } catch (e) {
    failRequest(`Not a valid request: ${e.message}`);
    return;
  }
  $("#curl-text").textContent = curlCommand(body);
  $("#curl-copy").textContent = "Copy";
  curlDialog.showModal();
};
$("#curl-copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("#curl-text").textContent);
    $("#curl-copy").textContent = "Copied";
  } catch {
    $("#curl-copy").textContent = "Select and copy by hand";
  }
};
$("#curl-close").onclick = () => curlDialog.close();
