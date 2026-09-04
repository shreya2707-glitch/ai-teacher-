"""
AI Teacher - Lesson State Machine
Core engine: PLAN -> EXPLAIN -> CHECK -> EVALUATE -> (ADAPT | ADVANCE) -> ... -> ASSESS -> REPORT

Requires: pip install anthropic --break-system-packages
Set ANTHROPIC_API_KEY as an environment variable before running.

This module is deliberately backend-agnostic w.r.t. RAG - retrieved_context is just
a string you pass in. Plug your Chroma/FAISS retrieval in before calling plan_lesson().
"""

import os
import json
import dataclasses
from typing import Optional
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"  # swap to whichever model you're using in the API


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Concept:
    name: str
    estimated_minutes: int
    depth: str                 # "intro" | "applied" | "technical"
    visual_type: str           # "diagram" | "equation" | "code" | "timeline" | "none"
    status: str = "not_started"   # not_started | in_progress | understood | struggling
    retry_count: int = 0
    last_misconception: Optional[str] = None


@dataclasses.dataclass
class LessonState:
    topic: str
    learner_level: str         # beginner | intermediate | advanced
    language: str              # e.g. "Hindi", "English", "Hinglish"
    time_budget_min: int
    retrieved_context: str = ""     # RAG chunks, empty string if topic-only mode
    concepts: list[Concept] = dataclasses.field(default_factory=list)
    current_idx: int = 0
    turn_count: int = 0
    transcript: list[dict] = dataclasses.field(default_factory=list)  # for demo/debug logging

    @property
    def current_concept(self) -> Optional[Concept]:
        if 0 <= self.current_idx < len(self.concepts):
            return self.concepts[self.current_idx]
        return None

    def log(self, stage: str, payload: dict):
        self.transcript.append({"turn": self.turn_count, "stage": stage, **payload})


MAX_RETRIES_PER_CONCEPT = 2


# ---------------------------------------------------------------------------
# LLM call helper - always asks for strict JSON, strips markdown fences defensively
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, max_tokens: int = 1000) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# 1. PLAN
# ---------------------------------------------------------------------------

def plan_lesson(topic: str, level: str, language: str, time_budget_min: int,
                 retrieved_context: str = "") -> LessonState:
    system = (
        "You are a curriculum planner for an AI tutor. Given a topic, learner level, "
        "available time, and optional source material, break the topic into an ordered "
        "list of teachable concepts. Respond with JSON ONLY, no preamble, no markdown "
        'fences. Schema: {"concepts": [{"name": str, "estimated_minutes": int, '
        '"depth": "intro"|"applied"|"technical", "visual_type": "diagram"|"equation"|'
        '"code"|"timeline"|"none"}]}. '
        "The sum of estimated_minutes should roughly match the time budget. "
        "Order concepts so each builds on the previous one."
    )
    user = (
        f"Topic: {topic}\n"
        f"Learner level: {level}\n"
        f"Time budget: {time_budget_min} minutes\n"
        f"Teaching language: {language}\n"
        f"Source material (may be empty if topic-only mode):\n{retrieved_context[:4000]}"
    )
    data = _call_llm(system, user)
    concepts = [Concept(**c) for c in data["concepts"]]
    state = LessonState(
        topic=topic, learner_level=level, language=language,
        time_budget_min=time_budget_min, retrieved_context=retrieved_context,
        concepts=concepts,
    )
    state.log("PLAN", {"concepts": [c.name for c in concepts]})
    return state


# ---------------------------------------------------------------------------
# 2. EXPLAIN
# ---------------------------------------------------------------------------

def explain_concept(state: LessonState, retry_hint: Optional[str] = None) -> dict:
    concept = state.current_concept
    system = (
        "You are a human-like AI teacher explaining ONE specific concept, scoped tightly - "
        "do not drift into other concepts. Respond with JSON ONLY. "
        'Schema: {"spoken_script": str, "on_screen_text": str, "worked_example": str, '
        '"visual_spec": str}. spoken_script is what the avatar will say aloud - natural, '
        "warm, conversational, in the target language. on_screen_text is short bullet-style "
        "text to display alongside. visual_spec briefly describes what diagram/equation/code "
        "should be shown (matching the concept's visual_type)."
    )
    retry_note = ""
    if retry_hint:
        retry_note = (
            f"\nIMPORTANT: The student struggled last time with this misconception: "
            f"'{retry_hint}'. Use a DIFFERENT analogy than a standard explanation, and "
            f"directly address this misconception."
        )
    user = (
        f"Concept: {concept.name}\n"
        f"Depth: {concept.depth}\n"
        f"Learner level: {state.learner_level}\n"
        f"Language: {state.language}\n"
        f"Suggested visual type: {concept.visual_type}\n"
        f"Relevant source material:\n{state.retrieved_context[:2000]}"
        f"{retry_note}"
    )
    data = _call_llm(system, user, max_tokens=800)
    concept.status = "in_progress"
    state.log("EXPLAIN", {"concept": concept.name, "retry": retry_hint is not None})
    return data


# ---------------------------------------------------------------------------
# 3. CHECK - generate a targeted question
# ---------------------------------------------------------------------------

def generate_check_question(state: LessonState) -> dict:
    concept = state.current_concept
    system = (
        "You are an AI teacher generating ONE check-for-understanding question about the "
        "concept just taught. Respond with JSON ONLY. "
        'Schema: {"question_type": "mcq"|"short_answer"|"explain_in_own_words"|'
        '"problem_solving", "question": str, "options": [str] (only if mcq, else empty '
        'list), "expected_answer_summary": str}. '
        "Vary question type based on depth: intro -> mcq or short_answer, "
        "technical -> problem_solving or explain_in_own_words. "
        "Write the question in the target language."
    )
    user = (
        f"Concept just taught: {concept.name}\n"
        f"Depth: {concept.depth}\n"
        f"Language: {state.language}"
    )
    data = _call_llm(system, user, max_tokens=400)
    state.log("CHECK", {"concept": concept.name, "question": data["question"]})
    return data


# ---------------------------------------------------------------------------
# 4. EVALUATE - the misconception-detection step
# ---------------------------------------------------------------------------

def evaluate_answer(state: LessonState, question: dict, student_answer: str) -> dict:
    concept = state.current_concept
    system = (
        "You are diagnosing a student's answer, not just grading it. Respond with JSON "
        'ONLY. Schema: {"classification": "correct"|"partially_correct"|"misconception"|'
        '"no_understanding", "misconception_type": str or null, "confidence": float '
        '(0-1), "feedback": str}. '
        "If classification is 'misconception', misconception_type should describe the "
        "specific WRONG mental model that likely produced this answer (e.g. 'treats "
        "current as a consumable resource, like water running out'), not just 'wrong'. "
        "feedback is a short, constructive, encouraging note in the target language - "
        "never mocking, always specific about what to fix."
    )
    user = (
        f"Concept: {concept.name}\n"
        f"Question asked: {question['question']}\n"
        f"Expected answer summary: {question['expected_answer_summary']}\n"
        f"Student's answer: {student_answer}\n"
        f"Language: {state.language}"
    )
    data = _call_llm(system, user, max_tokens=500)
    state.log("EVALUATE", {"concept": concept.name, "classification": data["classification"]})
    return data


# ---------------------------------------------------------------------------
# 5. ADAPT / ADVANCE - decision logic (no LLM call needed, pure state transition)
# ---------------------------------------------------------------------------

def adapt_or_advance(state: LessonState, evaluation: dict) -> str:
    """Returns 'advance', 'retry', or 'give_up_and_advance'."""
    concept = state.current_concept
    classification = evaluation["classification"]

    if classification in ("correct", "partially_correct") and evaluation["confidence"] >= 0.6:
        concept.status = "understood"
        state.current_idx += 1
        state.log("ADVANCE", {"concept": concept.name})
        return "advance"

    # misconception or no_understanding
    concept.retry_count += 1
    concept.last_misconception = evaluation.get("misconception_type")

    if concept.retry_count > MAX_RETRIES_PER_CONCEPT:
        concept.status = "struggling"
        state.current_idx += 1
        state.log("GIVE_UP_ADVANCE", {"concept": concept.name, "retries": concept.retry_count})
        return "give_up_and_advance"

    concept.status = "struggling"
    state.log("ADAPT_RETRY", {"concept": concept.name, "retry_count": concept.retry_count})
    return "retry"


# ---------------------------------------------------------------------------
# 6. ASSESS + REPORT
# ---------------------------------------------------------------------------

def generate_final_report(state: LessonState) -> dict:
    understood = [c.name for c in state.concepts if c.status == "understood"]
    struggling = [c.name for c in state.concepts if c.status == "struggling"]
    total = len(state.concepts)
    score_pct = round(100 * len(understood) / total) if total else 0

    system = (
        "You write a short, encouraging end-of-lesson report for a student, in the "
        "target language. Respond with JSON ONLY. "
        'Schema: {"summary": str, "recommendation": str, "suggested_next_topic": str}. '
        "Keep summary to 2-3 sentences. recommendation should be concrete and actionable."
    )
    user = (
        f"Topic: {state.topic}\n"
        f"Score: {score_pct}%\n"
        f"Strong areas: {', '.join(understood) or 'none yet'}\n"
        f"Needs improvement: {', '.join(struggling) or 'none'}\n"
        f"Language: {state.language}"
    )
    narrative = _call_llm(system, user, max_tokens=400)

    return {
        "topic": state.topic,
        "score_pct": score_pct,
        "strong_areas": understood,
        "needs_improvement": struggling,
        **narrative,
    }


# ---------------------------------------------------------------------------
# Example end-to-end run (text-only loop; swap input() for your API/frontend)
# ---------------------------------------------------------------------------

def run_lesson_cli(topic: str, level: str, language: str, time_budget_min: int,
                    retrieved_context: str = ""):
    state = plan_lesson(topic, level, language, time_budget_min, retrieved_context)
    print(f"\nLesson plan: {[c.name for c in state.concepts]}\n")

    while state.current_idx < len(state.concepts):
        state.turn_count += 1
        concept = state.current_concept
        retry_hint = concept.last_misconception if concept.retry_count > 0 else None

        explanation = explain_concept(state, retry_hint=retry_hint)
        print(f"\n--- {concept.name} ---")
        print(explanation["spoken_script"])
        print(f"[visual: {explanation['visual_spec']}]")

        question = generate_check_question(state)
        print(f"\nQ: {question['question']}")
        if question["options"]:
            for i, opt in enumerate(question["options"]):
                print(f"  {chr(97+i)}) {opt}")

        student_answer = input("Your answer: ")
        evaluation = evaluate_answer(state, question, student_answer)
        print(f"[{evaluation['classification']}] {evaluation['feedback']}")

        decision = adapt_or_advance(state, evaluation)
        if decision == "retry":
            print("(re-explaining with a different approach...)")

    report = generate_final_report(state)
    print("\n=== FINAL REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return state, report


if __name__ == "__main__":
    run_lesson_cli(
        topic="Ohm's Law",
        level="beginner",
        language="English",
        time_budget_min=20,
    )