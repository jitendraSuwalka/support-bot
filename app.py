"""
app.py — web interface for the support agent.

DESIGN NOTE
    A chat window is not what makes this project interesting; every LLM demo
    has one. What makes it interesting is that each answer carries its
    evidence and the record of the checks it passed. So the interface puts the
    verification strip and the audit trail alongside the reply rather than
    hiding them, and the visual language is borrowed from a compliance
    document: ink on paper, monospaced logs, semantic colour used only where
    it means something.

ARCHITECTURE NOTE
    This module does not modify agent.py. The agent prints its pipeline
    events to stdout; here that stream is captured and parsed. That keeps a
    single source of truth for agent behaviour, so all 52 tests in
    evaluate.py remain valid without change -- a wrapper cannot silently
    diverge from the thing it wraps.

Run:  python app.py      then open the local URL it prints
"""

import io
import re
import threading
from contextlib import redirect_stdout

import gradio as gr

import agent

# stdout is process-wide, so capture must be serialised. Gradio serves
# requests concurrently; without this lock two sessions would interleave
# their pipeline traces into each other's panels.
CAPTURE_LOCK = threading.Lock()

AUDIT_MARKER = "--- audit trail (raw evidence) ---"


# =========================================================
# RUNNING THE AGENT
# =========================================================


def run_with_trace(question, conversation, state):
    """Call the real agent, capturing its pipeline output."""
    buffer = io.StringIO()
    conversation.append({"role": "user", "content": question})
    with CAPTURE_LOCK, redirect_stdout(buffer):
        answer = agent.run_agent(conversation, state)
    return answer, buffer.getvalue()


def split_answer(answer):
    """Separate the customer-facing reply from its notes and evidence."""
    reply, evidence = answer, ""
    if AUDIT_MARKER in answer:
        reply, evidence = answer.split(AUDIT_MARKER, 1)

    notes = re.findall(r"\n\s*\[([^\]]+)\]\s*$", reply)
    reply = re.sub(r"\n\s*\[[^\]]+\]\s*$", "", reply).strip()
    return reply, notes, evidence.strip()


# =========================================================
# READING THE TRACE
# =========================================================


def build_status(trace, notes):
    """Turn pipeline output into a row of plain-language checks.

    Each entry answers a question a reviewer would actually ask: what did it
    look up, was every figure evidenced, did anything unsupported slip
    through, and did a human get involved.
    """
    chips = []

    tools = re.findall(r"\[tool\] (\w+)", trace)
    for name in dict.fromkeys(tools):
        label = {
            "search_knowledge_base": "Searched knowledge base",
            "look_up_client_query": "Looked up case status",
            "escalate_to_adviser": "Escalated to adviser",
        }.get(name, name)
        chips.append(("tool", label))

    if "[no candidates above floor" in trace:
        chips.append(("none", "No matching entry"))
    dropped = re.search(r"\[rerank dropped (\d+) of (\d+)", trace)
    if dropped:
        chips.append(("tool", f"Reranked, {dropped.group(1)} of {dropped.group(2)} discarded"))

    if "[citations: clean]" in trace:
        chips.append(("ok", "Figures verified"))
    elif "CITATION VIOLATIONS" in trace:
        state = "ok" if "clean after rewrite" in trace else "bad"
        chips.append((state, "Unsupported figure removed"))

    if "[verified: grounded]" in trace:
        chips.append(("ok", "Grounded in evidence"))
    elif "UNSUPPORTED CLAIMS" in trace:
        chips.append(("warn", "Rewritten after fact-check"))
    elif "VERIFICATION UNAVAILABLE" in trace:
        chips.append(("bad", "Fact-check unavailable"))

    if "POLICY: unresolved" in trace:
        chips.append(("warn", "Escalated by policy"))
    elif "ESCALATION LOGGED" in trace and "escalate_to_adviser" not in tools:
        chips.append(("warn", "Escalated to a human"))
    if "clarification allowed once" in trace:
        chips.append(("tool", "Clarification requested"))

    for note in notes:
        if note.startswith("WARNING"):
            chips.append(("bad", note.replace("WARNING: ", "")))

    if not chips:
        chips.append(("tool", "Answered without tools"))

    return "".join(f'<span class="chip chip-{kind}">{label}</span>' for kind, label in chips)


# =========================================================
# HANDLERS
# =========================================================

EMPTY_EVIDENCE = "_Evidence appears here once the agent has looked something up._"


def respond(question, history, conversation, state):
    question = (question or "").strip()
    if not question:
        return history, "", conversation, state, "", EMPTY_EVIDENCE, ""

    history = history + [{"role": "user", "content": question}]

    try:
        answer, trace = run_with_trace(question, conversation, state)
    except Exception as error:
        history.append(
            {
                "role": "assistant",
                "content": f"The agent could not complete that request. Details: `{error}`",
            }
        )
        return (
            history,
            "",
            conversation,
            state,
            '<span class="chip chip-bad">Request failed</span>',
            EMPTY_EVIDENCE,
            str(error),
        )

    reply, notes, evidence = split_answer(answer)
    history.append({"role": "assistant", "content": reply})

    evidence_md = f"```\n{evidence}\n```" if evidence else EMPTY_EVIDENCE
    trace_md = f"```\n{trace.strip()}\n```" if trace.strip() else "_No pipeline output._"

    return (history, "", conversation, state, build_status(trace, notes), evidence_md, trace_md)


def reset():
    return [], [], {"unclear_turns": 0}, "", EMPTY_EVIDENCE, ""


# =========================================================
# INTERFACE
# =========================================================

CSS = """
:root {
  --ink:      #16233b;
  --paper:    #f7f6f2;
  --slate:    #5b6980;
  --rule:     #d9d6cd;
  --ok:       #1f6b52;
  --warn:     #8a5a12;
  --bad:      #92321f;
}
.gradio-container { max-width: 1120px !important; }

#masthead { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 6px; }
#masthead h1 {
  font-size: 1.45rem; font-weight: 650; letter-spacing: -0.015em;
  color: var(--ink); margin: 0 0 4px 0;
}
#masthead p { color: var(--slate); margin: 0; font-size: 0.9rem; line-height: 1.5; }
#masthead .eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--slate); display: block; margin-bottom: 6px;
}

#statusbar { min-height: 34px; padding: 2px 0 6px 0; }
.chip {
  display: inline-block; margin: 3px 6px 3px 0; padding: 3px 10px;
  border: 1px solid var(--rule); border-radius: 2px; background: #fff;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.7rem; letter-spacing: 0.02em; color: var(--slate);
}
.chip-ok   { color: var(--ok);   border-color: var(--ok);   border-left-width: 3px; }
.chip-warn { color: var(--warn); border-color: var(--warn); border-left-width: 3px; }
.chip-bad  { color: var(--bad);  border-color: var(--bad);  border-left-width: 3px; }
.chip-none { color: var(--slate); border-style: dashed; }

#sidenote {
  border-left: 2px solid var(--rule); padding-left: 14px;
  color: var(--slate); font-size: 0.82rem; line-height: 1.6;
}
#sidenote strong { color: var(--ink); }

.panel-label {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--slate); border-bottom: 1px solid var(--rule);
  padding-bottom: 5px; margin: 18px 0 8px 0;
}
#evidence {
  max-height: 260px; overflow-y: auto;
  font-size: 0.8rem; color: var(--ink);
}
#evidence pre { background: #fff; border: 1px solid var(--rule); }

footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.stone,
).set(
    body_background_fill="#f7f6f2",
    block_border_width="1px",
    block_radius="2px",
    button_large_radius="2px",
    button_small_radius="2px",
    input_radius="2px",
)

EXAMPLES = [
    "What's the status of FW-1001?",
    "Should I move my pension into equities?",
    "Can I visit a branch on Sunday?",
    "Do you have a Manchester office?",
    "What happens if I'm dissatisfied?",
]

with gr.Blocks(title="Grounded Support Agent") as demo:
    conversation = gr.State([])
    policy_state = gr.State({"unclear_turns": 0})

    gr.HTML(
        '<div id="masthead">'
        '<span class="eyebrow">Retrieval &middot; verification &middot; escalation</span>'
        "<h1>Grounded support agent</h1>"
        "<p>Every answer is drawn from a sourced knowledge base, checked against "
        "its evidence, and escalated to a human when it cannot be supported. "
        "The checks each reply passed are shown below it.</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=380,
                show_label=False,
                placeholder="Ask about offices, complaints, appointments, "
                "or a case reference such as FW-1001.",
            )

            status = gr.HTML(elem_id="statusbar")

            with gr.Row(equal_height=True):
                question = gr.Textbox(
                    placeholder="Type your question",
                    show_label=False,
                    container=False,
                    scale=6,
                    min_width=220,
                    autofocus=True,
                )
                send = gr.Button("Send", variant="primary", scale=0, min_width=104)
                clear = gr.Button("Start over", scale=0, min_width=116)

            gr.Examples(examples=EXAMPLES, inputs=question, label="Try one")

        with gr.Column(scale=2):
            gr.HTML(
                '<div id="sidenote">'
                "<strong>What you are looking at.</strong> The agent may only "
                "state facts a tool returned in the current turn. Case statuses "
                "are quoted word for word and expire between turns. URLs and "
                "figures are matched against the evidence in code, not by "
                "judgement. Anything touching financial advice goes to a human."
                "<br><br>Fernhill Wealth is a fictional firm: client records, policies, "
                "addresses and phone numbers are all invented. A personal "
                "project."
                "</div>"
            )
            # Kept permanently visible rather than in an accordion: the evidence
            # is the point of the project, and a reviewer should never have to
            # click to see what a claim rests on.
            gr.HTML('<div class="panel-label">Evidence used</div>')
            evidence = gr.Markdown(EMPTY_EVIDENCE, elem_id="evidence")
            with gr.Accordion("Pipeline trace", open=False):
                trace = gr.Markdown("_Runs appear here._")

    outputs = [chatbot, question, conversation, policy_state, status, evidence, trace]
    question.submit(respond, [question, chatbot, conversation, policy_state], outputs)
    send.click(respond, [question, chatbot, conversation, policy_state], outputs)
    clear.click(
        reset,
        None,
        [chatbot, conversation, policy_state, status, evidence, trace],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(theme=THEME, css=CSS)
