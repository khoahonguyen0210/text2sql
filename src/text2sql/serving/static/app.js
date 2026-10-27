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

// Real answers from the fine-tuned run, played in the workspace until the visitor takes over.
const DEMOS = [
  ["concert_singer", "Show all countries and the number of singers in each country.",
    "SELECT country , count(*) FROM singer GROUP BY country",
    ["Country", "count(*)"], [["France", 4], ["Netherlands", 1], ["United States", 1]]],
  ["pets_1", "What is the average and maximum age for each pet type?",
    "SELECT pettype , avg(pet_age) , max(pet_age) FROM pets GROUP BY pettype",
    ["PetType", "avg(pet_age)", "max(pet_age)"], [["cat", 3.0, 3], ["dog", 1.5, 2]]],
  ["concert_singer", "Show the stadium names without any concert.",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
    ["Name"], [["Bayview Stadium"], ["Hampden Park"], ["Forthbank Stadium"], ["Gayfield Park"]]],
];

const ICONS = {
  battle_death: "⚔️", car_1: "🚗", concert_singer: "🎤", course_teach: "🎓",
  cre_Doc_Template_Mgt: "📄", dog_kennels: "🐕", employee_hire_evaluation: "💼",
  flight_2: "✈️", museum_visit: "🏛️", network_1: "💬", orchestra: "🎻", pets_1: "🐾",
  poker_player: "🃏", real_estate_properties: "🏠", singer: "🎵",
  student_transcripts_tracking: "📚", tvshow: "📺", voter_1: "🗳️", world_1: "🌍", wta_1: "🎾",
};

const SQL_WORDS = new Set(["SELECT", "FROM", "WHERE", "JOIN", "ON", "AS", "GROUP", "BY", "ORDER",
  "HAVING", "LIMIT", "NOT", "IN", "AND", "OR", "DESC", "ASC", "DISTINCT", "UNION", "INTERSECT",
  "EXCEPT", "COUNT", "AVG", "MAX", "MIN", "SUM", "LIKE", "BETWEEN"]);

const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let databases = [];
let selected = null;
let errorTimer = null;
let demoOn = true;

function prettyName(id) {
  return id.split("_").filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
}

const HUES = [355, 270, 215, 185, 20, 320, 240, 340];

function hueOf(text) {
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.charCodeAt(0)) % 9973;
  return HUES[hash % HUES.length];
}

function iconOf(id) {
  return ICONS[id] || "🗄️";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  $("db-icon").textContent = iconOf(selected.db_id);
  $("db-name").textContent = prettyName(selected.db_id);
  $("db-tables").textContent =
    `${count} ${count === 1 ? "table" : "tables"} · ${selected.tables.join(", ")}`;
  $("question").placeholder = `Type a question about the ${prettyName(selected.db_id)} data`;
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

// Empties steps 2 and 3, so the window never shows SQL for another question.
function clearAnswer() {
  $("sql").value = "";
  $("safety").textContent = "";
  $("meta").textContent = "";
  $("step-sql").classList.remove("on");
  syncButtons(false);
  clearRows();
}

// Run and Copy stay visible, and work only when there is SQL to use.
function syncButtons(hasSql = $("sql").value.trim() !== "") {
  $("run").disabled = !hasSql;
  $("copy").disabled = !hasSql;
}

function clearRows() {
  $("result").replaceChildren();
  $("result-meta").textContent = "";
  $("rows-empty").hidden = false;
  $("step-rows").classList.remove("on");
}

// The visitor takes over the workspace: the example stops and the real tool shows.
function stopDemo() {
  if (!demoOn) return;
  demoOn = false;
  $("demo-tag").hidden = true;
  $("demo-sql").hidden = true;
  $("sql").hidden = false;
  $("step-sql").classList.remove("typing");
  $("question").value = "";
  clearAnswer();
}

function pickDatabase(id) {
  stopDemo();
  if (selected && selected.db_id !== id) clearAnswer();
  selectDatabase(id);
}

function makeTile(db) {
  const tile = el("button", "tile");
  tile.type = "button";
  tile.dataset.db = db.db_id;
  tile.style.setProperty("--h", hueOf(db.db_id));
  tile.setAttribute("aria-pressed", "false");
  tile.append(
    el("span", "tile-icon", iconOf(db.db_id)),
    el("span", "tile-tag", "SELECTED"),
    el("span", "tile-name", prettyName(db.db_id)),
    el("span", "tile-sub", `${db.tables.length} tables · ${db.tables.slice(0, 3).join(", ")}`),
  );
  tile.addEventListener("click", () => {
    pickDatabase(db.db_id);
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
  body.append(
    el("span", "rank-q", question),
    el("span", "rank-db", `${iconOf(dbId)}  ${prettyName(dbId)}`),
  );
  card.append(el("span", "rank", String(index + 1)), body);
  card.addEventListener("click", () => {
    pickDatabase(dbId);
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
  stopDemo();
  showError("");
  const question = $("question").value.trim();
  if (!question) {
    showError("Type a question first.");
    $("question").focus();
    return;
  }
  clearAnswer();
  setBusy(true);
  try {
    const data = await call("/v1/generate-sql", { question, db_id: selected.db_id });
    $("sql").value = data.sql;
    const badge = $("safety");
    badge.className = "badge " + (data.safety.ok ? "safe" : "unsafe");
    badge.textContent = data.safety.ok ? "✓ Safe" : `✕ Not safe (${data.safety.reason})`;
    $("meta").textContent = `${(data.latency_ms / 1000).toFixed(1)} s`;
    $("step-sql").classList.add("on");
    syncButtons();
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
  rows.forEach((row, index) => {
    const tr = body.insertRow();
    tr.style.setProperty("--i", Math.min(index, 20));
    for (const value of row) tr.insertCell().textContent = value === null ? "NULL" : String(value);
  });
  $("rows-empty").hidden = true;
  $("step-rows").classList.add("on");
}

async function runSql() {
  showError("");
  $("run").disabled = true;
  try {
    const data = await call("/v1/run-sql", { db_id: selected.db_id, sql: $("sql").value });
    fillTable(data.columns, data.rows);
    const rows = `${data.row_count} ${data.row_count === 1 ? "row" : "rows"}`;
    const more = data.truncated ? " (first rows only)" : "";
    $("result-meta").textContent = `${rows} · ${data.elapsed_ms.toFixed(1)} ms${more}`;
  } catch (err) {
    showError(err.message);
  } finally {
    syncButtons();
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

// Each typing helper returns false once the visitor has taken over, so the demo stops mid-way.
async function typeInto(node, text, ms, prop = "textContent") {
  if (calm) {
    node[prop] += text;
    return demoOn;
  }
  for (const ch of text) {
    if (!demoOn) return false;
    node[prop] += ch;
    await sleep(ms);
  }
  return demoOn;
}

async function typeSql(node, sql) {
  for (const token of sql.split(/(\s+|\(|\))/).filter(Boolean)) {
    const word = token.toUpperCase();
    const span = el("span", SQL_WORDS.has(word) ? "kw" : /^\d/.test(token) ? "num" : "");
    node.append(span);
    if (!(await typeInto(span, token, 22))) return false;
  }
  return demoOn;
}

async function playDemo() {
  for (let i = 0; demoOn; i = (i + 1) % DEMOS.length) {
    const [dbId, question, sql, columns, rows] = DEMOS[i];
    selectDatabase(dbId);
    $("question").value = "";
    $("demo-sql").replaceChildren();
    $("step-sql").classList.remove("on");
    syncButtons(false);
    clearRows();

    if (!(await typeInto($("question"), question, 28, "value"))) return;
    if (!demoOn) return;
    await sleep(350);

    $("step-sql").classList.add("on", "typing");
    if (!(await typeSql($("demo-sql"), sql))) return;
    $("step-sql").classList.remove("typing");
    syncButtons(true);
    await sleep(500);
    if (!demoOn) return;

    // Show the Run button being pressed, so the visitor sees where the rows come from.
    $("run").classList.add("press");
    await sleep(450);
    $("run").classList.remove("press");
    if (!demoOn) return;

    fillTable(columns, rows);
    $("result-meta").textContent = `${rows.length} rows`;
    await sleep(4500);
  }
}

function countUp(node) {
  const target = Number(node.dataset.count);
  const decimals = (node.dataset.count.split(".")[1] || "").length;
  const show = (value) => {
    node.textContent =
      `${node.dataset.prefix || ""}${value.toFixed(decimals)}${node.dataset.suffix || ""}`;
  };
  if (calm) return show(target);
  const start = performance.now();
  const frame = (now) => {
    const t = Math.min((now - start) / 1400, 1);
    show(target * (1 - Math.pow(1 - t, 3)));
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function setupReveal() {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("in");
      const number = entry.target.querySelector("[data-count]");
      if (number) countUp(number);
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.2 });
  document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
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
  if (DEMOS.every(([db]) => known.has(db))) playDemo();
  else stopDemo();
}

$("ask").addEventListener("submit", writeSql);
$("run").addEventListener("click", runSql);
$("copy").addEventListener("click", copySql);
$("sql").addEventListener("input", () => {
  // Hand-edited SQL has not been checked yet; the API checks it again when it runs.
  $("safety").textContent = "";
  $("step-sql").classList.toggle("on", $("sql").value.trim() !== "");
  syncButtons();
});
$("app").addEventListener("pointerdown", stopDemo);
$("app").addEventListener("focusin", stopDemo);
window.addEventListener("scroll", () => $("nav").classList.toggle("solid", window.scrollY > 10),
  { passive: true });
init().catch(() => showError("Could not load the list of databases."));
checkReady();
setupReveal();
