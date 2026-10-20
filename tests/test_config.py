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
        ("SUPPORT_BOT_TOP_K", "9"),  # exceeds the default of 5 candidates
    ],
)
def test_invalid_values_are_rejected_loudly(name, value):
    with pytest.raises(ValueError):
        Settings.from_env({name: value})


def test_describe_lists_every_setting_for_the_audit_record():
    described = Settings().describe()
    assert described["main_model"] == "claude-sonnet-5-5"
    assert set(described) >= {"main_effort", "max_steps", "relevance_floor", "require_evidence"}
