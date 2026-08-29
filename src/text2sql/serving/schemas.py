from pydantic import BaseModel, ConfigDict, Field

DB_ID = r"^[a-z0-9_]{1,64}$"
REQUEST_ID = r"^[A-Za-z0-9_-]{1,64}$"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)
    db_id: str = Field(pattern=DB_ID)
    request_id: str | None = Field(default=None, pattern=REQUEST_ID)


class SafetyInfo(BaseModel):
    ok: bool
    reason: str


class ModelInfo(BaseModel):
    id: str
    revision: str
    adapter: str | None
    prompt_version: str


class GenerateResponse(BaseModel):
    request_id: str
    db_id: str
    sql: str
    status: str
    safety: SafetyInfo
    model: ModelInfo
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_id: str = Field(pattern=DB_ID)
    sql: str = Field(min_length=1, max_length=4000)


class RunResponse(BaseModel):
    db_id: str
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    elapsed_ms: float


class DatabaseInfo(BaseModel):
    db_id: str
    tables: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
