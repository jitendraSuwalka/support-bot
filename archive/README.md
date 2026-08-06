# Build history

The agent was built incrementally, and these are the intermediate versions.
They are kept because the progression shows how each guard came to exist —
almost all of them were added in response to a specific failure observed while
testing the version before.

| File | What it added |
|---|---|
| `step2.py` | A single API call |
| `step3.py` | Conversation memory |
| `step4.py` | First tool, and the tool-use request/result cycle |
| `step6.py` | The agent loop, three tools, system-prompt guardrails |
| `step7.py` | Forced-fresh tool calls; expiring volatile history |
| `step8.py` | Query expansion, BM25 scoring, verbatim quoting, groundedness check |
| `step9.py` | Deterministic citation guard, LLM reranking |

The working version is `agent.py` in the repository root. Nothing here is
imported by it, and these files are not covered by the test suite.
