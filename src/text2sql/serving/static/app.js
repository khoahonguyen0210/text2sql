const $ = (id) => document.getElementById(id);
let databases = [];

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
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

function showTables() {
  const db = databases.find((d) => d.db_id === $("db").value);
  $("tables").textContent = db ? "Tables: " + db.tables.join(", ") : "";
}

async function loadDatabases() {
  databases = await call("/v1/databases");
  for (const db of databases) {
    const option = document.createElement("option");
    option.value = db.db_id;
    option.textContent = db.db_id;
    $("db").appendChild(option);
  }
  showTables();
}

async function writeSql() {
  showError("");
  const question = $("question").value.trim();
  if (!question) {
    showError("Type a question first.");
    return;
  }
  $("write").disabled = true;
  $("write").textContent = "Writing...";
  try {
    const data = await call("/v1/generate-sql", { question, db_id: $("db").value });
    $("sql").value = data.sql;
    const safety = data.safety.ok ? "safe to run" : "not safe to run (" + data.safety.reason + ")";
    $("meta").textContent =
      "Written in " + (data.latency_ms / 1000).toFixed(1) + " s. The query is " + safety + ".";
    $("sql-card").hidden = false;
    $("result-card").hidden = true;
  } catch (err) {
    showError(err.message);
  } finally {
    $("write").disabled = false;
    $("write").textContent = "Write SQL";
  }
}

function fillTable(columns, rows) {
  const table = $("result");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const name of columns) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) {
      tr.insertCell().textContent = value === null ? "NULL" : String(value);
    }
  }
}

async function runSql() {
  showError("");
  $("run").disabled = true;
  try {
    const data = await call("/v1/run-sql", { db_id: $("db").value, sql: $("sql").value });
    fillTable(data.columns, data.rows);
    const more = data.truncated ? " Only the first " + data.row_count + " rows are shown." : "";
    $("result-meta").textContent =
      data.row_count + " rows in " + data.elapsed_ms.toFixed(1) + " ms." + more;
    $("result-card").hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    $("run").disabled = false;
  }
}

$("db").addEventListener("change", showTables);
$("write").addEventListener("click", writeSql);
$("run").addEventListener("click", runSql);
loadDatabases().catch(() => showError("Could not load the list of databases."));
