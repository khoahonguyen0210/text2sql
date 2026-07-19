from tests.conftest import TOY_TABLES
from text2sql.config import generation_config
from text2sql.data.schema import schema_from_spider
from text2sql.models.prompt import PromptConfig, build_messages
from text2sql.paths import CONFIGS_DIR

GOLD = "SELECT count(*) FROM singer WHERE age > 40"


def test_prompt_holds_schema_and_question_but_no_gold_sql():
    messages = build_messages("How many singers are older than 40?",
                              schema_from_spider(TOY_TABLES), PromptConfig())
    text = " ".join(m["content"] for m in messages)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "CREATE TABLE singer" in text
    assert "older than 40" in text
    assert GOLD not in text


def test_prompt_hash_changes_with_settings():
    assert PromptConfig().hash() == PromptConfig().hash()
    assert PromptConfig().hash() != PromptConfig(include_foreign_keys=False).hash()
    assert PromptConfig().hash() != PromptConfig(schema_style="compact").hash()


def test_base_config_loads():
    cfg = generation_config(CONFIGS_DIR / "base.toml")
    assert cfg.model.revision == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    assert cfg.decoding.do_sample is False
    assert cfg.prompt == PromptConfig()
