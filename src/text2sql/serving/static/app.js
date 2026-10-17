const $ = (id) => document.getElementById(id);

// Real Spider dev questions that the fine-tuned model answers correctly.
const EXAMPLES = [
  ["dog_kennels", "Find the states where both owners and professionals live."],
  ["world_1", "Which region is the city Kabul located in?"],
  ["car_1", "How many models does each car maker produce? List maker full name, id and the number."],
  ["concert_singer", "Show name, country, age for all singers ordered by age from the oldest to the youngest."],
  ["pets_1", "Find the maximum weight for each type of pet. List the maximum weight and pet type."],
  ["museum_visit", "Find the id and name of the museum that has the most staff members?"],
  ["flight_2", 'List all airline names and their abbreviations in "USA".'],
  ["orchestra", 'What are the maximum and minimum share of performances whose type is not "Live final".'],
  ["real_estate_properties", "What is the feature type name of feature AirCon?"],
  ["employee_hire_evaluation", "What is the number of employees from each city?"],
];

let databases = [];
let selected = null;
let errorTimer = null;

function prettyName(id) {
  return id.split("_").filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
}

const HUES = [355, 270, 215, 185, 20, 320, 240, 340];

function hueOf(text) {
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.charCodeAt(0)) % 9973;
  return HUES[hash % HUES.length];
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showError(message) {
  clearTimeout(errorTimer);
  $("error").textContent = message;
  $("error").hidden = !message;
  if (message) errorTimer = setTimeout(() => ($("error").hidden = true), 8000);
}

async function call(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ? data.error.message : "The request failed.");
  }
  return data;
}

function selectDatabase(id) {
  selected = databases.find((d) => d.db_id === id) || databases[0];
  const count = selected.tables.length;
  $("db-eyebrow").textContent = `Database · ${count} ${count === 1 ? "table" : "tables"}`;
  $("db-title").textContent = prettyName(selected.db_id);
  $("tables").textContent = selected.tables.join(" · ");
  $("hero").style.setProperty("--h", hueOf(selected.db_id));
  const track = $("db-track");
  for (const tile of track.children) {
    const on = tile.dataset.db === selected.db_id;
    tile.setAttribute("aria-pressed", String(on));
    const hidden = tile.offsetLeft < track.scrollLeft ||
      tile.offsetLeft + tile.offsetWidth > track.scrollLeft + track.clientWidth;
    if (on && hidden) {
      track.scrollTo({ left: tile.offsetLeft - (track.clientWidth - tile.offsetWidth) / 2 });
    }
  }
}

function makeTile(db) {
  const tile = el("button", "tile");
  tile.type = "button";
  tile.dataset.db = db.db_id;
  tile.style.setProperty("--h", hueOf(db.db_id));
  tile.setAttribute("aria-pressed", "false");
  tile.append(
    el("span", "tile-tag", "SELECTED"),
    el("span", "tile-name", prettyName(db.db_id)),
    el("span", "tile-sub", `${db.tables.length} tables · ${db.tables.slice(0, 3).join(", ")}`),
  );
  tile.addEventListener("click", () => {
    selectDatabase(db.db_id);
    window.scrollTo({ top: 0 });
    $("question").focus({ preventScroll: true });
  });
  return tile;
}

function makeExample([dbId, question], index) {
  const card = el("button", "rank-card");
  card.type = "button";
  card.setAttribute("aria-label", `Example ${index + 1}: ${question}`);
  const body = el("span", "rank-body");
  body.style.setProperty("--h", hueOf(dbId));
  body.append(el("span", "rank-q", question), el("span", "rank-db", prettyName(dbId)));
  card.append(el("span", "rank", String(index + 1)), body);
  card.addEventListener("click", () => {
    selectDatabase(dbId);
    $("question").value = question;
    window.scrollTo({ top: 0 });
    writeSql();
  });
  return card;
}

function setupRails() {
  for (const rail of document.querySelectorAll(".rail")) {
    const track = rail.querySelector(".rail-track");
    for (const arrow of rail.querySelectorAll(".rail-arrow")) {
      const direction = arrow.classList.contains("next") ? 1 : -1;
      arrow.addEventListener("click", () =>
        track.scrollBy({ left: direction * track.clientWidth * 0.8, behavior: "smooth" }));
    }
  }
}

function setBusy(busy) {
  $("write").disabled = busy;
  $("question").disabled = busy;
  $("progress").hidden = !busy;
  const old = $("write").querySelector(".spinner");
  if (old) old.remove();
  clearInterval(setBusy.timer);
  if (!busy) {
    $("write-label").textContent = "Write SQL";
    return;
  }
  $("write").prepend(el("span", "spinner"));
  const start = Date.now();
  const tick = () => {
    $("write-label").textContent = `Writing ${Math.floor((Date.now() - start) / 1000)} s`;
  };
  tick();
  setBusy.timer = setInterval(tick, 1000);
}

async function writeSql(event) {
  if (event) event.preventDefault();
  showError("");
  const question = $("question").value.trim();
  if (!question) {
    showError("Type a question first.");
    $("question").focus();
    return;
  }
  setBusy(true);
  try {
    const data = await call("/v1/generate-sql", { question, db_id: selected.db_id });
    $("answer-db").textContent = `Your SQL · ${prettyName(data.db_id)}`;
    $("answer-title").textContent = question;
    $("sql").value = data.sql;
    const badge = $("safety");
    badge.className = "badge " + (data.safety.ok ? "safe" : "unsafe");
    badge.textContent = data.safety.ok ? "Safe to run" : `Not safe to run (${data.safety.reason})`;
    $("meta").textContent = `Written in ${(data.latency_ms / 1000).toFixed(1)} s`;
    $("result-wrap").hidden = true;
    $("answer").hidden = false;
    $("answer").scrollIntoView({ block: "start" });
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

function fillTable(columns, rows) {
  const table = $("result");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const name of columns) head.appendChild(el("th", "", name));
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) tr.insertCell().textContent = value === null ? "NULL" : String(value);
  }
}

async function runSql() {
  showError("");
  $("run").disabled = true;
  try {
    const data = await call("/v1/run-sql", { db_id: selected.db_id, sql: $("sql").value });
    fillTable(data.columns, data.rows);
    const rows = `${data.row_count} ${data.row_count === 1 ? "row" : "rows"}`;
    const more = data.truncated ? ` Only the first ${data.row_count} rows are shown.` : "";
    $("result-meta").textContent = `${rows} in ${data.elapsed_ms.toFixed(1)} ms.${more}`;
    $("result-wrap").hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    $("run").disabled = false;
  }
}

async function copySql() {
  try {
    await navigator.clipboard.writeText($("sql").value);
    $("copy-label").textContent = "Copied";
    setTimeout(() => ($("copy-label").textContent = "Copy SQL"), 1500);
  } catch {
    showError("Could not copy the SQL.");
  }
}

async function checkReady() {
  try {
    const data = await call("/ready");
    $("status-text").textContent = data.model.adapter ? "Fine-tuned model ready" : "Base model ready";
    $("status").classList.add("ready");
  } catch {
    setTimeout(checkReady, 3000);
  }
}

async function init() {
  setupRails();
  databases = await call("/v1/databases");
  $("db-track").append(...databases.map(makeTile));
  const known = new Set(databases.map((d) => d.db_id));
  $("example-track").append(...EXAMPLES.filter(([db]) => known.has(db)).map(makeExample));
  selectDatabase(known.has("concert_singer") ? "concert_singer" : databases[0].db_id);
}

$("ask").addEventListener("submit", writeSql);
$("run").addEventListener("click", runSql);
$("copy").addEventListener("click", copySql);
window.addEventListener("scroll", () => $("nav").classList.toggle("solid", window.scrollY > 10),
  { passive: true });
init().catch(() => showError("Could not load the list of databases."));
checkReady();
