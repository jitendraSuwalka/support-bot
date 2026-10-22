"""Settings come from the environment, are validated, and have safe defaults."""

import pytest

from config import Settings


def test_defaults_are_the_current_pinned_models():
    settings = Settings()
    assert settings.main_model == "claude-sonnet-5-5"
    assert settings.helper_model == "claude-haiku-5-5"


def test_every_setting_can_be_overridden_from_the_environment():
    env = {
        "SUPPORT_BOT_MAIN_MODEL": "claude-opus-5-5",
        "SUPPORT_BOT_MAIN_EFFORT": "medium",
        "SUPPORT_BOT_MAX_STEPS": "3",
        "SUPPORT_BOT_RELEVANCE_FLOOR": "0.3",
        "SUPPORT_BOT_USE_RERANK": "false",
        "SUPPORT_BOT_REQUIRE_EVIDENCE": "0",
    }
    settings = Settings.from_env(env)
    assert settings.main_model == "claude-opus-5-5"
    assert settings.main_effort == "medium"
    assert settings.max_steps == 3
    assert settings.relevance_floor == 0.3
    assert settings.use_rerank is False
    assert settings.require_evidence is False
    assert settings.helper_model == "claude-haiku-5-5"  # untouched default


def test_retrieval_settings_default_to_the_local_model_and_an_on_disk_index():
    settings = Settings()
    assert settings.retrieval_mode == "hybrid"
    assert settings.relevance_floor == 0.15 and settings.dense_floor == 0.50
    assert settings.candidates == 6 and settings.top_k == 3
    assert settings.embedder == "local" and settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.index_path == ".index" and settings.model_cache == ".models"
    assert settings.qdrant_url == ""  # local mode unless a server is named
    overridden = Settings.from_env(
        {"SUPPORT_BOT_EMBEDDER": "hash", "SUPPORT_BOT_QDRANT_URL": "http://localhost:6333"}
    )
    assert overridden.embedder == "hash" and overridden.qdrant_url == "http://localhost:6333"


def test_blank_values_fall_back_to_defaults():
    settings = Settings.from_env({"SUPPORT_BOT_MAIN_MODEL": "   "})
    assert settings.main_model == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SUPPORT_BOT_MAIN_EFFORT", "turbo"),
        ("SUPPORT_BOT_MAX_STEPS", "0"),
        ("SUPPORT_BOT_RELEVANCE_FLOOR", "1.5"),
        ("SUPPORT_BOT_USE_RERANK", "maybe"),
        ("SUPPORT_BOT_TOP_K", "9"),  # exceeds the default of 6 candidates
        ("SUPPORT_BOT_EMBEDDER", "openai"),
        ("SUPPORT_BOT_DENSE_FLOOR", "-0.1"),
        ("SUPPORT_BOT_RETRIEVAL_MODE", "sparse"),
    ],
)
def test_invalid_values_are_rejected_loudly(name, value):
    with pytest.raises(ValueError):
        Settings.from_env({name: value})


def test_a_blank_path_set_in_code_is_rejected():
    with pytest.raises(ValueError, match="index_path"):
        Settings(index_path="   ").validate()


def test_describe_lists_every_setting_for_the_audit_record():
    described = Settings().describe()
    assert described["main_model"] == "claude-sonnet-5-5"
    assert set(described) >= {"main_effort", "max_steps", "relevance_floor", "require_evidence"}
