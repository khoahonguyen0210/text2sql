import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from text2sql.data.schema import Schema
from text2sql.evaluation.extract import extract_sql
from text2sql.serving.schemas import (
    DatabaseInfo,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    RunRequest,
    RunResponse,
    SafetyInfo,
)
from text2sql.sql_safety import check_sql
from text2sql.sqlite_exec import execute_readonly

STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 16384
    max_question_chars: int = 500
    generate_per_minute: int = 20
    run_per_minute: int = 60
    query_timeout_s: float = 5.0
    max_rows: int = 100
    max_cell_chars: int = 200


@dataclass
class Generated:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class Backend:
    """Anything that turns (question, schema) into model text. The real one wraps the model."""

    info: ModelInfo

    def generate(self, question: str, schema: Schema) -> Generated:
        raise NotImplementedError


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            calls = self.calls[key]
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= self.per_minute:
                return False
            calls.append(now)
            return True


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(status_code=status, detail={"code": code, "message": message},
                         headers=headers)


def _cell(value, max_chars: int):
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def create_app(
    backend_factory: Callable[[], Backend],
    schemas: dict[str, Schema],
    db_paths: dict[str, Path],
    limits: Limits | None = None,
) -> FastAPI:
    """db_paths is the allowlist: only these databases can be used, by name."""
    limits = limits or Limits()
    gen_limiter = RateLimiter(limits.generate_per_minute)
    run_limiter = RateLimiter(limits.run_per_minute)
    gen_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend_factory()  # loaded once, not per request
        yield
        app.state.backend = None

    app = FastAPI(title="Text-to-SQL API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def limit_and_harden(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > limits.max_body_bytes):
            return _error(413, "body_too_large", "The request body is too large.")
        if request.method == "POST" and not length:
            return _error(411, "length_required", "Send a Content-Length header.")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if not request.url.path.startswith(("/docs", "/openapi")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
            )
        return response

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse({"error": exc.detail}, exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError):
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
        return _error(422, "invalid_request", f"Check these fields: {fields}.")

    @app.exception_handler(Exception)
    async def internal(_: Request, __: Exception):
        return _error(500, "internal_error", "Something went wrong on the server.")

    def client_key(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def schema_for(db_id: str) -> Schema:
        if db_id not in db_paths:
            raise ApiError(404, "unknown_database", f"No database named '{db_id}'.")
        return schemas[db_id]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if getattr(request.app.state, "backend", None) is None:
            raise ApiError(503, "not_ready", "The model is not loaded yet.")
        return {"status": "ready", "model": request.app.state.backend.info}

    @app.get("/v1/databases", response_model=list[DatabaseInfo])
    def databases():
        return [DatabaseInfo(db_id=d, tables=[t.name for t in schemas[d].tables])
                for d in sorted(db_paths)]

    @app.post("/v1/generate-sql", response_model=GenerateResponse)
    def generate_sql(body: GenerateRequest, request: Request):
        if not gen_limiter.allow(client_key(request)):
            raise ApiError(429, "rate_limited", "Too many requests. Try again in a minute.",
                           {"Retry-After": "60"})
        schema = schema_for(body.db_id)
        backend: Backend = request.app.state.backend
        with gen_lock:  # one generation at a time; the model is not shared safely
            out = backend.generate(body.question, schema)
        sql, status = extract_sql(out.text)
        safety = check_sql(sql) if sql else None
        return GenerateResponse(
            request_id=body.request_id or uuid.uuid4().hex[:16],
            db_id=body.db_id,
            sql=sql,
            status=status,
            safety=SafetyInfo(ok=bool(safety and safety.ok),
                              reason=safety.reason if safety else "no_sql"),
            model=backend.info,
            latency_ms=out.latency_ms,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
        )

    @app.post("/v1/run-sql", response_model=RunResponse)
    def run_sql(body: RunRequest, request: Request):
        if not run_limiter.allow(client_key(request)):
            raise ApiError(429, "rate_limited", "Too many requests. Try again in a minute.",
                           {"Retry-After": "60"})
        schema_for(body.db_id)
        safety = check_sql(body.sql)
        if not safety.ok:
            raise ApiError(400, "unsafe_sql", f"Only one read-only SELECT is allowed "
                                              f"({safety.reason}).")
        result = execute_readonly(db_paths[body.db_id], body.sql,
                                  timeout_s=limits.query_timeout_s, max_rows=limits.max_rows)
        if result.status == "timeout":
            raise ApiError(408, "query_timeout", "The query took too long and was stopped.")
        if result.status == "denied":
            raise ApiError(400, "unsafe_sql", "The database refused this query.")
        if result.status != "ok":
            message = result.error if not result.error.startswith("open failed") else (
                "The database could not be opened.")
            raise ApiError(400, "query_error", message[:300])
        rows = [[_cell(v, limits.max_cell_chars) for v in row] for row in result.rows]
        return RunResponse(db_id=body.db_id, columns=result.columns, rows=rows,
                           row_count=len(rows), truncated=result.truncated,
                           elapsed_ms=result.elapsed_ms)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status)
