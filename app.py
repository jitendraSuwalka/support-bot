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
    This module does not modify agent.py. The agent returns an AgentResult —
    the customer-facing answer, the notes about what the guards did, the raw
    evidence and a structured trace of pipeline events — and this module only
    presents it. Nothing is parsed out of printed text, so the interface
    cannot drift from the agent and needs no locking: each request's trace
    belongs to that request alone.

Run:  python app.py      then open the local URL it prints
"""

import gradio as gr

import agent

TOOL_LABELS = {
    "search_knowledge_base": "Searched knowledge base",
    "look_up_client_query": "Looked up case status",
    "escalate_to_adviser": "Escalated to adviser",
}


# =========================================================
# READING THE RESULT
# =========================================================


def build_status(result):
    """Turn the trace into a row of plain-language checks.

    Each entry answers a question a reviewer would actually ask: what did it
    look up, was every figure evidenced, did anything unsupported slip
    through, and did a human get involved.
    """
    kinds = result.trace.kinds()
    chips = []

    for name in dict.fromkeys(result.tools_used):
        chips.append(("tool", TOOL_LABELS.get(name, name)))

    if "evidence_guard" in kinds:
        chips.append(("warn", "Sent back for a lookup"))
    if "no_candidates" in kinds or "rerank_rejected_all" in kinds:
        chips.append(("none", "No matching entry"))
    for event in result.trace.find("rerank"):
        chips.append(
            ("tool", f"Reranked, {event.data['dropped']} of {event.data['total']} discarded")
        )

    if "citation_violation" in kinds:
        state = "bad" if "citations_remaining" in kinds else "ok"
        chips.append((state, "Unsupported figure removed"))
    elif "citations_clean" in kinds:
        chips.append(("ok", "Figures verified"))

    if "verified" in kinds:
        chips.append(("ok", "Grounded in evidence"))
    elif "unsupported" in kinds:
        chips.append(("warn", "Rewritten after fact-check"))
    elif "verification_unavailable" in kinds:
        chips.append(("bad", "Fact-check unavailable"))

    if "policy_escalate" in kinds:
        chips.append(("warn", "Escalated by policy"))
    if "policy_clarify" in kinds:
        chips.append(("tool", "Clarification requested"))

    for note in result.notes:
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
    conversation.append({"role": "user", "content": question})

    try:
        result = agent.run_agent(conversation, state)
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

    history.append({"role": "assistant", "content": result.answer})

    evidence_text = "\n\n".join(result.evidence)
    evidence_md = f"```\n{evidence_text}\n```" if evidence_text else EMPTY_EVIDENCE
    trace_text = result.trace.render()
    trace_md = f"```\n{trace_text}\n```" if trace_text else "_No pipeline events._"

    return (history, "", conversation, state, build_status(result), evidence_md, trace_md)


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

# analytics_enabled=False: Gradio reports usage to api.gradio.app by default.
# A support tool for a regulated firm sends nothing anywhere it does not have to.
with gr.Blocks(title="Grounded Support Agent", analytics_enabled=False) as demo:
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
    # Open the document index before serving, so the first question is not
    # the one that pays for loading (and, on a new machine, downloading) the
    # embedding model. The trace line it logs is printed, as there is no run.
    print("Preparing the document index (first run: downloads the embedding model, about 70 MB)")
    agent.get_retriever()
    demo.queue(default_concurrency_limit=4).launch(theme=THEME, css=CSS)
