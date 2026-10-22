"""
Runtime settings, read once from the environment.

Every tunable the agent has lives here, with a documented default, so a
deployment can change behaviour without editing code and a reviewer can see
every knob in one place. Values come from environment variables prefixed
SUPPORT_BOT_ (see .env.example); anything unset keeps its default.

Model IDs are pinned snapshots on the Claude API, so a deployment that records
them (the agent attaches them to every result) can be reproduced later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

ENV_PREFIX = "SUPPORT_BOT_"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EMBEDDERS = ("local", "hash")
RETRIEVAL_MODES = ("hybrid", "lexical", "dense")
AGENT_IMPLS = ("loop", "graph")


@dataclass(frozen=True)
class Settings:
    # Models. The main model writes answers and calls tools; the helper model
    # does the cheap, bounded jobs (query expansion, reranking, fact-checking).
    main_model: str = "claude-sonnet-5-5"
    helper_model: str = "claude-haiku-5-5"

    # Effort trades thoroughness for latency and cost. A support chat is
    # latency-sensitive and the questions are narrow, so both start at "low";
    # raise it if an evaluation shows quality improving.
    main_effort: str = "low"
    helper_effort: str = "low"

    # max_tokens is a hard ceiling on thinking plus reply, so leave headroom.
    main_max_tokens: int = 4096
    helper_max_tokens: int = 2048

    # Agent loop. "loop" is the hand-written loop in agent.py; "graph" is the
    # same turn as a LangGraph state graph (agent_graph.py). Both share every
    # model call, tool and guard; the scenario tests run both.
    agent_impl: str = "loop"
    max_steps: int = 8

    # Retrieval. Search is hybrid: BM25 and the vector index, fused by
    # reciprocal rank; "lexical" or "dense" runs one method alone (used to
    # compare them). A candidate must clear a floor on at least one side.
    # The lexical floor is a share of the query's BM25 ceiling and is
    # deliberately loose; the dense floor is a cosine similarity, set from
    # measurement (off-topic questions peak at about 0.53, on-topic start at
    # 0.68). Reranking does the real filtering: retrieve `candidates` widely,
    # keep `top_k` after reranking.
    retrieval_mode: str = "hybrid"
    relevance_floor: float = 0.15
    dense_floor: float = 0.60
    candidates: int = 6
    top_k: int = 3

    # Guards. Each can be switched off independently; a fully guarded answer
    # costs several model calls, which is right for a regulated firm and wrong
    # for most products, so the trade-off is explicit rather than baked in.
    use_rerank: bool = True
    verify_answers: bool = True
    check_citations: bool = True
    require_evidence: bool = True

    # Embeddings and the vector index. "local" runs a small ONNX model on the
    # CPU, downloaded to model_cache on first use; "hash" is a deterministic
    # stand-in with no model, for tests and offline CI. The index lives in
    # index_path unless qdrant_url points at a Qdrant server, in which case
    # the same code talks to the server instead.
    embedder: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache: str = ".models"
    index_path: str = ".index"
    qdrant_url: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Build settings from SUPPORT_BOT_* variables, validating each one."""
        source = os.environ if env is None else env
        values = {}
        for field in fields(cls):
            raw = source.get(ENV_PREFIX + field.name.upper())
            if raw is None or raw.strip() == "":
                continue
            values[field.name] = _parse(field.name, raw.strip(), field.type)
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        for name in ("main_effort", "helper_effort"):
            if getattr(self, name) not in EFFORT_LEVELS:
                raise ValueError(f"{name} must be one of {EFFORT_LEVELS}")
        for name in ("main_max_tokens", "helper_max_tokens", "max_steps", "candidates", "top_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise ValueError("relevance_floor must be between 0 and 1")
        if not 0.0 <= self.dense_floor <= 1.0:
            raise ValueError("dense_floor must be between 0 and 1")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")
        if self.top_k > self.candidates:
            raise ValueError("top_k cannot exceed candidates")
        if self.agent_impl not in AGENT_IMPLS:
            raise ValueError(f"agent_impl must be one of {AGENT_IMPLS}")
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"embedder must be one of {EMBEDDERS}")
        for name in ("embedding_model", "model_cache", "index_path"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")

    def describe(self) -> dict[str, object]:
        """A plain dictionary, attached to every result for auditability."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _parse(name: str, raw: str, kind: str | type) -> object:
    kind = str(kind)
    if kind in ("bool", "<class 'bool'>"):
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name}: expected true/false, got {raw!r}")
    if kind in ("int", "<class 'int'>"):
        return int(raw)
    if kind in ("float", "<class 'float'>"):
        return float(raw)
    return raw
