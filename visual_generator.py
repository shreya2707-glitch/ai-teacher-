"""
AI Teacher - Subject-Aware Visual Generation
Generates the on-screen visual for a concept, based on the visual_type the PLAN
step already assigned ("diagram" | "equation" | "code" | "timeline" | "none").

Design choice: rather than one generic image-generation call for everything, each
visual_type has its own renderer, because "subject-aware visual explanation" is an
explicit judged criterion (Section 10) - a generic AI image for an equation looks
worse than an actual rendered equation, and judges notice the difference.

Renderers:
    equation  -> matplotlib + LaTeX rendering (crisp, exact, no hallucinated symbols)
    code      -> syntax-highlighted code block image (Pygments) + optional execution output
    diagram   -> structured description -> rendered via Mermaid (flowchart/process diagrams)
    timeline  -> structured events -> rendered via Mermaid timeline
    none      -> skipped, on_screen_text alone carries the slide

Requires:
    pip install matplotlib pygments requests --break-system-packages
    Mermaid rendering uses the free mermaid.ink API (no key needed) - swap for a
    self-hosted mermaid-cli if you want to avoid the external dependency on demo day.
"""

import os
import base64
import json
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

OUTPUT_DIR = "./generated_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _call_llm(system: str, user: str, max_tokens: int = 600) -> dict:
    response = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# EQUATION - matplotlib LaTeX rendering
# ---------------------------------------------------------------------------

def render_equation(concept_name: str, visual_spec: str, out_name: str = "equation.png") -> str:
    """
    Asks the LLM for the exact LaTeX for the concept's key equation, then renders
    it crisply with matplotlib (no hallucinated symbols - LLM only supplies the
    LaTeX string, matplotlib does the actual rendering).
    """
    system = (
        "Return JSON ONLY: {\"latex\": str, \"label\": str}. "
        "latex is a single valid LaTeX math expression (no $ delimiters, no \\begin{equation}) "
        "for the concept's core formula. label is a short caption."
    )
    data = _call_llm(system, f"Concept: {concept_name}\nContext: {visual_spec}")

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.6, f"${data['latex']}$", fontsize=24, ha="center", va="center")
    ax.text(0.5, 0.15, data["label"], fontsize=11, ha="center", va="center", color="gray")
    ax.axis("off")

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CODE - syntax-highlighted snippet, optionally with real execution output
# ---------------------------------------------------------------------------

def render_code(concept_name: str, visual_spec: str, language: str = "python",
                 run_it: bool = False, out_name: str = "code.html") -> dict:
    """
    Generates a short illustrative code snippet + (optionally) actually executes it
    to get real output rather than an LLM-guessed output - matters for the
    "execution flow" requirement in Section 10 and avoids showing wrong output.
    Returns {"html_path": ..., "code": ..., "output": ... or None}.
    """
    system = (
        "Return JSON ONLY: {\"code\": str}. "
        "code is a short (5-15 line), runnable, illustrative code snippet in the "
        "requested language demonstrating the concept. No markdown fences inside the value."
    )
    data = _call_llm(system, f"Concept: {concept_name}\nLanguage: {language}\nContext: {visual_spec}")
    code = data["code"]

    output = None
    if run_it and language.lower() == "python":
        output = _run_python_snippet(code)

    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter

    lexer = get_lexer_by_name(language)
    formatter = HtmlFormatter(style="monokai", noclasses=True)
    html_code = highlight(code, lexer, formatter)

    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w") as f:
        f.write(html_code)
        if output:
            f.write(f"<pre style='color:#0f0;background:#000;padding:8px'>{output}</pre>")

    return {"html_path": out_path, "code": code, "output": output}


def _run_python_snippet(code: str, timeout_sec: float = 5.0) -> str:
    """
    Executes untrusted-ish LLM-generated code in a subprocess with a timeout.
    For a hackathon demo this is acceptable since the code is short and
    self-generated, but do NOT expose run_it=True on a public-facing endpoint
    without sandboxing (e.g. a container with no network/filesystem access).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["python3", "-c", code], capture_output=True, text=True, timeout=timeout_sec
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "(execution timed out)"


# ---------------------------------------------------------------------------
# DIAGRAM - Mermaid flowchart/process diagram via mermaid.ink
# ---------------------------------------------------------------------------

def render_diagram(concept_name: str, visual_spec: str, out_name: str = "diagram.png") -> str:
    """
    Asks the LLM for a Mermaid flowchart definition, renders it as PNG via the
    free mermaid.ink API. Good for processes, architectures, biological cycles, etc.
    """
    system = (
        "Return JSON ONLY: {\"mermaid\": str}. "
        "mermaid is a valid Mermaid.js flowchart definition (start with 'flowchart TD' "
        "or 'flowchart LR') visually explaining the concept's process or structure. "
        "Keep it to 4-8 nodes - simple and readable, not cluttered."
    )
    data = _call_llm(system, f"Concept: {concept_name}\nContext: {visual_spec}")
    return _render_mermaid(data["mermaid"], out_name)


def render_timeline(concept_name: str, visual_spec: str, out_name: str = "timeline.png") -> str:
    """For History-type concepts - renders a Mermaid timeline diagram."""
    system = (
        "Return JSON ONLY: {\"mermaid\": str}. "
        "mermaid is a valid Mermaid.js 'timeline' definition (start with 'timeline') "
        "with a title and 4-8 dated/ordered events relevant to the concept."
    )
    data = _call_llm(system, f"Concept: {concept_name}\nContext: {visual_spec}")
    return _render_mermaid(data["mermaid"], out_name)


def _render_mermaid(mermaid_code: str, out_name: str) -> str:
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


# ---------------------------------------------------------------------------
# Dispatcher - single entry point, matches visual_type from lesson_engine.Concept
# ---------------------------------------------------------------------------

def generate_visual(concept_name: str, visual_type: str, visual_spec: str,
                     code_language: str = "python", run_code: bool = False) -> dict:
    """
    concept_name, visual_type, visual_spec come straight from lesson_engine.py -
    visual_type from Concept.visual_type, visual_spec from explain_concept()'s output.

    Returns {"type": visual_type, "path"/"html_path": ..., ...} - shape varies
    slightly by type since equation/diagram/timeline are images and code is HTML.
    """
    visual_type = visual_type.lower()

    if visual_type == "equation":
        path = render_equation(concept_name, visual_spec)
        return {"type": "equation", "path": path}

    if visual_type == "code":
        result = render_code(concept_name, visual_spec, language=code_language, run_it=run_code)
        return {"type": "code", **result}

    if visual_type == "diagram":
        path = render_diagram(concept_name, visual_spec)
        return {"type": "diagram", "path": path}

    if visual_type == "timeline":
        path = render_timeline(concept_name, visual_spec)
        return {"type": "timeline", "path": path}

    return {"type": "none"}


if __name__ == "__main__":
    # smoke tests - requires ANTHROPIC_API_KEY, and network access for mermaid.ink
    print(generate_visual("Ohm's Law", "equation", "V = IR, voltage current resistance relationship"))
    print(generate_visual("Water Cycle", "diagram", "evaporation, condensation, precipitation, collection"))
    print(generate_visual("For loops", "code", "iterate 1 to 5 and print squares", code_language="python", run_code=True))