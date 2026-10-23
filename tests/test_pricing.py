"""The cost model: every call priced from its tokens, and an unknown model never free."""

import pytest

import agent
from config import Settings
from pricing import (
    LONG_CONTEXT_TOKENS,
    PRICES,
    Usage,
    call_cost,
    price_for,
    turn_cost,
    usage_from_events,
)


def test_the_configured_models_are_priced():
    # A model change in config.py without a price would make every turn unpriced.
    defaults = Settings()
    assert defaults.main_model in PRICES and defaults.helper_model in PRICES


def test_a_call_is_priced_per_token_type():
    # Sonnet 5.5: $2 input, $2.50 cache write, $0.10 cache read, $10 output per million.
    usage = Usage(
        model="claude-sonnet-5-5",
        input_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert call_cost(usage) == pytest.approx(2.00 + 2.50 + 0.10 + 10.00)
    assert call_cost(Usage("claude-haiku-5-5", input_tokens=3000, output_tokens=400)) == (
        pytest.approx((3000 * 0.10 + 400 * 0.50) / 1_000_000)
    )


def test_a_long_prompt_is_priced_at_the_long_context_rate_where_there_is_one():
    long = LONG_CONTEXT_TOKENS + 1
    assert price_for("claude-haiku-5-5", long).input == 0.50
    assert price_for("claude-haiku-5-5", LONG_CONTEXT_TOKENS).input == 0.10
    # Cache reads count towards the prompt's length.
    usage = Usage("claude-haiku-5-5", input_tokens=10, cache_read_tokens=long)
    assert (
        usage.prompt_tokens == long + 10
        and price_for(usage.model, usage.prompt_tokens).input == 0.50
    )
    assert price_for("claude-sonnet-5-5", long).input == 2.00  # no separate tier


def test_an_unknown_model_is_unpriced_not_free():
    assert call_cost(Usage("claude-future-9", input_tokens=100)) is None
    assert call_cost(Usage(None, input_tokens=100)) is None
    cost = turn_cost([Usage("claude-sonnet-5-5", input_tokens=10), Usage("claude-future-9", 5)])
    assert cost.usd is None and cost.unpriced_models == ("claude-future-9",)
    assert cost.input_tokens == 15 and cost.calls == 2


def test_a_turn_adds_its_calls_up():
    cost = turn_cost(
        [
            Usage(
                "claude-sonnet-5-5", input_tokens=2000, output_tokens=100, cache_read_tokens=3000
            ),
            Usage("claude-haiku-5-5", input_tokens=500, output_tokens=50),
        ]
    )
    expected = (2000 * 2.00 + 100 * 10.00 + 3000 * 0.10 + 500 * 0.10 + 50 * 0.50) / 1_000_000
    assert cost.usd == pytest.approx(expected)
    assert cost.as_dict()["usd"] == round(expected, 6)
    assert (cost.input_tokens, cost.output_tokens, cost.cache_read_tokens) == (2500, 150, 3000)
    assert turn_cost([]).usd == 0 and turn_cost([]).calls == 0


def test_a_result_prices_itself_from_its_trace():
    trace = agent.Trace()
    trace.log(
        "usage", "u", role="main", model="claude-sonnet-5-5", input_tokens=1000, output_tokens=100
    )
    trace.log(
        "usage", "u", role="helper", model="claude-haiku-5-5", input_tokens=400, output_tokens=40
    )
    trace.log("tool", "not a usage event")
    assert len(usage_from_events(trace.find("usage"))) == 2
    result = agent.AgentResult(
        answer="x", notes=[], evidence=[], trace=trace, escalated=False, tools_used=[], models={}
    )
    cost = result.cost()
    assert cost.calls == 2 and cost.usd == pytest.approx(
        (1000 * 2 + 100 * 10 + 400 * 0.1 + 40 * 0.5) / 1e6
    )
