"""
What a turn costs, from the tokens its model calls used.

Prices are US dollars per million tokens, from the Claude API pricing page
(https://platform.claude.com/docs/en/about-claude/pricing), checked on
23 October 2026. Each model call is priced on its own: plain input, input
written to the prompt cache (five-minute cache), input read from the cache,
and output. Haiku 5.5 charges more for a prompt over 100,000 tokens; no call
this agent makes comes near that, and a call that did would be priced at
the long-context rate rather than the short one.

A model the table does not know is never priced at zero: its cost is None,
the turn's total is None, and the log says which model was missing, so a
model change that outruns the table shows up as a gap rather than as a
system that suddenly became free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

PRICES_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICES_CHECKED = "2026-10-23"
LONG_CONTEXT_TOKENS = 100_000


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    cache_write: float  # five-minute cache
    cache_read: float
    output: float


PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(input=2.00, cache_write=2.50, cache_read=0.10, output=10.00),
    "claude-haiku-5-5": Price(input=0.10, cache_write=0.125, cache_read=0.01, output=0.50),
}
LONG_CONTEXT_PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(input=0.50, cache_write=0.625, cache_read=0.05, output=2.50),
}


@dataclass(frozen=True)
class Usage:
    """The tokens one model call used, as the API reports them."""

    model: str | None
    input_tokens: int = 0  # not cached
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


def price_for(model: str | None, prompt_tokens: int = 0) -> Price | None:
    if model is None:
        return None
    if prompt_tokens > LONG_CONTEXT_TOKENS and model in LONG_CONTEXT_PRICES:
        return LONG_CONTEXT_PRICES[model]
    return PRICES.get(model)


def call_cost(usage: Usage) -> float | None:
    """The dollar cost of one call, or None for a model the table does not know."""
    price = price_for(usage.model, usage.prompt_tokens)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.cache_write_tokens * price.cache_write
        + usage.cache_read_tokens * price.cache_read
        + usage.output_tokens * price.output
    ) / 1_000_000


@dataclass(frozen=True)
class TurnCost:
    usd: float | None  # None when any call's model is unpriced
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    calls: int
    unpriced_models: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "usd": None if self.usd is None else round(self.usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "calls": self.calls,
            "unpriced_models": list(self.unpriced_models),
        }


def turn_cost(calls: Iterable[Usage]) -> TurnCost:
    calls = list(calls)
    costs = [call_cost(call) for call in calls]
    unpriced = tuple(
        sorted({str(call.model) for call, cost in zip(calls, costs, strict=True) if cost is None})
    )
    return TurnCost(
        usd=None if unpriced else sum(cost for cost in costs if cost is not None),
        input_tokens=sum(call.input_tokens for call in calls),
        output_tokens=sum(call.output_tokens for call in calls),
        cache_write_tokens=sum(call.cache_write_tokens for call in calls),
        cache_read_tokens=sum(call.cache_read_tokens for call in calls),
        calls=len(calls),
        unpriced_models=unpriced,
    )


def usage_from_events(events) -> list[Usage]:
    """The Usage of every model call, from a trace's "usage" events."""
    return [
        Usage(
            model=event.data.get("model"),
            input_tokens=event.data.get("input_tokens", 0) or 0,
            output_tokens=event.data.get("output_tokens", 0) or 0,
            cache_write_tokens=event.data.get("cache_write_tokens", 0) or 0,
            cache_read_tokens=event.data.get("cache_read_tokens", 0) or 0,
        )
        for event in events
    ]
