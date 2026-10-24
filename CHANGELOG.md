# Changelog

Every release of the support agent, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions
follow [Semantic Versioning](https://semver.org/). The release workflow takes a
version's notes from this file, adds the headline numbers from the committed
reports, and refuses to publish if the tag, this file and the code disagree on
the version.

## [Unreleased]

### Changed

- A rewrite or a greeting sends the tools with `tool_choice` set to `none`,
  so it reads the cached tools and system prompt like every other main-model
  call, instead of writing a cache entry of its own.

### Added

- `python -m eval.live --workers 4` runs four cases at a time, with the same
  rows in the same order, so a full evaluation takes about a quarter of the
  wait. Searches of the local index take a lock, which the HTTP service's
  thread pool needed too.

- Every evaluation row records the numbers and links in the final reply that
  nothing supports, checked after the turn whichever guards were on, so an
  ablation can show what reaches the customer; and, when the fact-checker
  forced a rewrite, the claims it flagged, the draft it read and the evidence.

### Fixed

- The release notes ran on past their own entry when the next heading had no
  brackets: the 2.0.0 notes carried the 1.0.0 entry before their measured
  numbers. Any second-level heading now ends an entry.

## [2.0.0] - 2026-10-24

The agent rebuilt as a measured system: a document corpus and hybrid
retrieval in place of hand-written snippets, a labelled evaluation set that
the README's results tables and headline numbers are rendered from, guards
enforced in code, and the agent served over HTTP from a container that CI
builds and tests.

### Added

- **A fictional firm to answer for.** Fernhill Wealth, a UK wealth manager with
  eighteen documents (fees, ISAs, pensions, complaints, contact details and
  more) on the reserved `.example` domain, every public figure reviewed against
  gov.uk, the FSCS and the Financial Ombudsman, and phone numbers from Ofcom's
  drama ranges, so the agent can never send a customer to a real place.
- **Hybrid retrieval.** A heading-aware chunker, local ONNX embeddings
  (`BAAI/bge-small-en-v1.5` through fastembed), a Qdrant vector store synced by
  content hash, BM25 over the same chunks, and reciprocal rank fusion, with
  BM25's first choice guaranteed a place, then an LLM rerank.
- **The agent as a graph.** The same turn as a LangGraph state graph, selectable
  by a setting, with its Mermaid diagram generated and held to the README by a
  test; the scenario tests run on both implementations.
- **An evaluation harness.** A labelled set of 169 customer questions,
  reviewed by a separate AI reviewer briefed as a compliance officer and read
  again against the first run; retrieval metrics per search mode with a sweep
  of the fusion constant; an end-to-end scorer of behaviour, escalation,
  required facts, guards, latency, tokens and cost; and the README's tables
  and headline rendered from the committed reports and held to them by a
  test, so those numbers can only be measured.
- **Behaviours that end a turn in code.** Clarify once, then escalate; decline
  what is out of scope; answer a greeting without tools; look up a case
  reference only if the customer gave it.
- **An HTTP service.** FastAPI with `POST /chat`, `DELETE /sessions/{id}`,
  `/health`, `/ready` and OpenAPI docs; server-issued conversation ids, limits
  as settings, one turn at a time per conversation, and a request id on every
  response.
- **Observability and cost.** One JSON log line per turn and per request, with
  behaviour, guards, latency, tokens and cost, and the customer's words
  recorded only as a length and a hash; the cost of every answer from a price
  table with its source and date; prompt caching, measured off and on.
- **A container.** The model and index built into the image, a non-root user,
  a Docker health check and an allow-listed build context; CI builds it and
  runs fifteen smoke checks on every push.
- **Engineering foundations.** Pinned dependencies and a cross-platform
  lockfile, a pytest suite with a scripted fake model that needs no network,
  ruff for lint and format, CI on Python 3.11 to 3.13 and Windows, Dependabot
  for actions and the base image, and every runtime setting read from the
  environment with validation.
- **Releases.** This changelog, and a workflow that turns a version tag into a
  tested container image on the GitHub Container Registry and a GitHub
  Release.

### Changed

- The knowledge base, paraphrased in 1.0.0 from a real firm's public pages, is
  replaced by the fictional firm's corpus.
- The agent returns a structured result (answer, behaviour, sources, guards,
  cost) that the web interface presents, instead of output the interface
  parsed.
- Escalation rules narrowed to what a person must act on, so questions about
  how things work are answered from the documents rather than handed off.

### Fixed

Each found by a test or by the evaluation, and described in the README's
*Failure modes found in testing*:

- A citation check that read two numbers separated by a comma as one, and
  one that flagged a number the customer had written themselves.
- A fact-check rewrite that was never itself citation-checked.
- A tokenizer that dropped the last word of every sentence, and plural folding
  that turned "branches" into "branche".
- A dense-similarity floor set from two probes that dropped 8 of 131
  answering passages and filtered nothing.
- Telemetry sent by the web framework and the embedding runtime.
- A prompt injection answered with a poem, a case reference completed from a
  guess, greetings with a pleasantry escalated, and the agent's own
  instructions counted as customer turns.

Found by an independent AI review of this release before it was published:

- Three replies the model writes outside the evidence path (the rewrite after
  a policy escalation, a clarifying question, a greeting) were never checked
  for numbers and links; a last-line check now replaces a failing one with a
  fixed reply.
- A phone number the customer wrote was accepted as evidence, and a case
  reference was matched inside a longer one (FW-10021 authorised FW-1002).
- A turn that failed after a tool call left the conversation unusable, and
  the web interface showed the exception to the customer.
- The exception's text reached the log; conversation ids had 122 random bits,
  not 128.
- Tokens per case left out cached tokens, and a retrieval miss was counted at
  ten results where the README read it at six.
- The service could rewrite its own code inside the container, and the
  suggested run command published an unauthenticated port to the network.

## 1.0.0 - 2026-08-06

The first version, never tagged: a tool-using support agent with a Gradio
interface, a citation guard, an independent fact-check, policy-enforced
escalation and 52 offline checks, answering from short passages paraphrased
from a real firm's public web pages.

[2.0.0]: https://github.com/jitendraSuwalka/support-bot/commits/v2.0.0
