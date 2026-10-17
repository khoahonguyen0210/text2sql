import pytest
from fastapi.testclient import TestClient

from tests.conftest import TOY_TABLES
from text2sql.data.schema import schema_from_spider
from text2sql.manifest import sha256_file
from text2sql.serving.app import Backend, Generated, Limits, create_app
from text2sql.serving.schemas import ModelInfo


class FakeBackend(Backend):
    info = ModelInfo(id="fake", revision="0", adapter=None, prompt_version="v1")

    def __init__(self, text: str):
        self.text = text

    def generate(self, question, schema):
        return Generated(self.text, 10, 5, 1.0)


@pytest.fixture
def make_client(toy_db):
    def make(text: str = "SELECT count(*) FROM singer", limits: Limits | None = None):
        backend = FakeBackend(text)
        app = create_app(lambda: backend, {"toy": schema_from_spider(TOY_TABLES)},
                         {"toy": toy_db}, limits)
        return TestClient(app, raise_server_exceptions=False)

    return make


def test_health_and_ready(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["status"] == "ready"


def test_generate_returns_sql_and_safety(make_client):
    with make_client("```sql\nSELECT count(*) FROM singer\n```") as client:
        r = client.post("/v1/generate-sql", json={"question": "How many singers?", "db_id": "toy"})
        assert r.status_code == 200
        body = r.json()
        assert body["sql"] == "SELECT count(*) FROM singer"
        assert body["status"] == "fenced"
        assert body["safety"] == {"ok": True, "reason": ""}


def test_generated_write_is_flagged_not_run(make_client, toy_db):
    before = sha256_file(toy_db)
    with make_client("DROP TABLE singer") as client:
        r = client.post("/v1/generate-sql", json={"question": "x", "db_id": "toy"})
        assert r.json()["safety"]["ok"] is False
    assert sha256_file(toy_db) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "db_id": "toy"},
        {"question": "x" * 501, "db_id": "toy"},
        {"question": "x", "db_id": "../toy"},
        {"question": "x", "db_id": "C:/data/toy.sqlite"},
        {"question": "x", "db_id": "toy", "extra": 1},
        {"question": "x", "db_id": "toy", "request_id": "bad id!"},
    ],
)
def test_bad_input_is_rejected(make_client, payload):
    with make_client() as client:
        r = client.post("/v1/generate-sql", json=payload)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"
        assert "x" * 50 not in r.text


def test_database_name_with_capitals(toy_db):
    app = create_app(lambda: FakeBackend("SELECT count(*) FROM singer"),
                     {"cre_Toy_DB": schema_from_spider(TOY_TABLES)}, {"cre_Toy_DB": toy_db})
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/v1/generate-sql", json={"question": "x", "db_id": "cre_Toy_DB"})
        assert r.status_code == 200


def test_unknown_database(make_client):
    with make_client() as client:
        r = client.post("/v1/generate-sql", json={"question": "x", "db_id": "other"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_database"


def test_run_sql(make_client):
    with make_client() as client:
        sql = "SELECT name FROM singer WHERE country = 'France' ORDER BY name"
        r = client.post("/v1/run-sql", json={"db_id": "toy", "sql": sql})
        assert r.status_code == 200
        assert r.json()["rows"] == [["Justin Brown"], ["Rose White"]]


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM singer", "SELECT 1; DROP TABLE singer", "PRAGMA table_info(singer)",
     "ATTACH DATABASE 'x.db' AS x", "SELECT load_extension('x')"],
)
def test_run_sql_blocks_unsafe_queries(make_client, toy_db, sql):
    before = sha256_file(toy_db)
    with make_client() as client:
        r = client.post("/v1/run-sql", json={"db_id": "toy", "sql": sql})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "unsafe_sql"
    assert sha256_file(toy_db) == before


def test_run_sql_time_limit(make_client):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    with make_client(limits=Limits(query_timeout_s=0.3)) as client:
        r = client.post("/v1/run-sql", json={"db_id": "toy", "sql": sql})
        assert r.status_code == 408


def test_run_sql_row_limit(make_client):
    with make_client(limits=Limits(max_rows=2)) as client:
        r = client.post("/v1/run-sql", json={"db_id": "toy", "sql": "SELECT * FROM singer"})
        assert r.json()["row_count"] == 2
        assert r.json()["truncated"] is True


def test_errors_do_not_leak_paths(make_client, toy_db):
    with make_client() as client:
        r = client.post("/v1/run-sql", json={"db_id": "toy", "sql": "SELECT nope FROM singer"})
        assert r.status_code == 400
        assert str(toy_db.parent) not in r.text


def test_body_size_limit(make_client):
    with make_client(limits=Limits(max_body_bytes=100)) as client:
        r = client.post("/v1/generate-sql", content=b"{" + b" " * 200 + b"}",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413


def test_rate_limit(make_client):
    with make_client(limits=Limits(generate_per_minute=2)) as client:
        payload = {"question": "x", "db_id": "toy"}
        codes = [client.post("/v1/generate-sql", json=payload).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_security_headers_and_page(make_client):
    with make_client() as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in r.headers["Content-Security-Policy"]
