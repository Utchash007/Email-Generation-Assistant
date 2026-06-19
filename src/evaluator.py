import json
import re
from typing import List, Dict, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from src.config import get_instructor_client, MODEL_A
from pydantic import BaseModel, Field
import instructor

# Schema for LLM-as-a-judge Fact Recall checklist
class FactCheckResult(BaseModel):
    recalled_facts_count: int = Field(description="The number of facts successfully included in the email.")
    total_facts_count: int = Field(description="The total number of facts that were supposed to be included.")
    analysis: str = Field(description="Brief analysis detailing which facts were present or missing.")

# Schema for LLM-as-a-judge Tone Adherence score
class ToneAdherenceResult(BaseModel):
    score_1_to_5: int = Field(description="Tone rating: 5=Perfect, 4=Very Good, 3=Average, 2=Poor, 1=Completely Off.")
    reasoning: str = Field(description="Detailed explanation justifying the tone score based on vocabulary, style, and structure.")

# Schema for LLM-as-a-judge Structure & Professionalism check
class StructureResult(BaseModel):
    layout_score_1_to_5: int = Field(description="Structure rating: 5=Excellent layout, 4=Good layout, 3=Mediocre, 2=Messy, 1=Unusable.")
    has_meta_text: bool = Field(description="True if the output contains conversational LLM text, e.g. 'Sure, here is your email:'")
    reasoning: str = Field(description="Explanation of email formatting, greetings, closings, and flow.")


class FactRecallMetric(BaseMetric):
    """
    Custom Metric 1: Fact Recall Metric.
    Measures the ratio of key input facts successfully included in the generated email.
    Uses LLM-as-a-Judge for semantic alignment of the facts.
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.client = get_instructor_client(mode=instructor.Mode.MD_JSON)

    def measure(self, test_case: LLMTestCase) -> float:
        # Extra input data is passed to the test_case.context or test_case.input
        # In our case, the input query is passed as test_case.input, output as test_case.actual_output.
        # We also need the original facts. We pass the list of facts serialized in test_case.context.
        facts = test_case.context if test_case.context else []
        email = test_case.actual_output

        if not facts:
            self.score = 1.0
            self.reason = "No key facts provided to recall."
            return self.score

        prompt = (
            f"You are an objective auditor. Your job is to verify how many of the required key facts are included "
            f"in the generated email. Treat facts as recalled if they are present either verbatim or semantically.\n\n"
            f"Required Key Facts:\n"
            + "\n".join(f"- {fact}" for fact in facts)
            + f"\n\nGenerated Email:\n{email}\n\n"
            f"Evaluate and output the results matching the required output structure."
        )

        try:
            # Query OpenRouter using instructor for structured auditing
            result: FactCheckResult = self.client.chat.completions.create(
                model=MODEL_A,
                response_model=FactCheckResult,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # Calculate the score ratio
            numerator = result.recalled_facts_count
            denominator = len(facts)
            self.score = min(float(numerator / denominator), 1.0)
            self.reason = f"Recalled {numerator} out of {denominator} facts. Analysis: {result.analysis}"
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error during fact check evaluation: {str(e)}"

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Fact Recall Metric"


class ToneAdherenceMetric(BaseMetric):
    """
    Custom Metric 2: Tone Adherence Metric.
    Measures how accurately the email matches the target tone (formal, casual, urgent, empathetic).
    Uses LLM-as-a-Judge with a standardized rubric.
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.client = get_instructor_client(mode=instructor.Mode.MD_JSON)

    def measure(self, test_case: LLMTestCase) -> float:
        # Target tone is expected to be passed as test_case.input or in test_case.context.
        # We will retrieve target tone from the test_case.expected_output (e.g. format: "Tone: {tone}")
        # or pass it explicitly. We'll store tone in the test_case.input.
        
        target_tone = "professional"
        # Extract tone from input string if stored in format "Intent: ... | Tone: {tone}"
        match = re.search(r"Tone:\s*(\w+)", test_case.input, re.IGNORECASE)
        if match:
            target_tone = match.group(1).lower()

        email = test_case.actual_output

        prompt = (
            f"You are a tone analyst. Rate how well the generated email matches the target tone: '{target_tone}'.\n\n"
            f"Rubric:\n"
            f"- 5 (Perfect): Vocabulary, structure, and pacing perfectly match the '{target_tone}' tone. Feels natural and appropriate.\n"
            f"- 4 (Very Good): Mostly aligns, with minor choices that could be adjusted.\n"
            f"- 3 (Average): The tone is somewhat present, but style feels mismatched in several places.\n"
            f"- 2 (Poor): Major deviations. For example, a formal request sounds too casual, or empathetic sounds cold.\n"
            f"- 1 (Completely Off): The email is the complete opposite of the requested tone.\n\n"
            f"Generated Email:\n{email}\n\n"
            f"Evaluate and output the results matching the required output structure."
        )

        try:
            result: ToneAdherenceResult = self.client.chat.completions.create(
                model=MODEL_A,
                response_model=ToneAdherenceResult,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # Normalize 1-5 score to 0.0 - 1.0 range
            self.score = float((result.score_1_to_5 - 1) / 4)
            self.reason = f"Tone Adherence score: {result.score_1_to_5}/5. Reasoning: {result.reasoning}"
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error during tone check evaluation: {str(e)}"

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Tone Adherence Metric"


class StructurePolishMetric(BaseMetric):
    """
    Custom Metric 3: Structure & Professionalism Metric.
    Checks email layout, paragraph layout, greeting/sign-off presence, and absence of LLM intro/outro fluff.
    Combines programmatic checks (40% weight) and LLM layout quality check (60% weight).
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.client = get_instructor_client(mode=instructor.Mode.MD_JSON)

    def measure(self, test_case: LLMTestCase) -> float:
        email = test_case.actual_output

        # --- 1. Programmatic Checks (40% weight) ---
        # A. Check greeting (Dear, Hi, Hey, Hello, To, Good morning, Good afternoon, etc.)
        greeting_patterns = [
            r"^(dear|hi|hey|hello|to|good\s+morning|good\s+afternoon|good\s+evening)\b"
        ]
        has_greeting = any(re.search(pat, email.strip(), re.IGNORECASE) for pat in greeting_patterns)

        # B. Check sign-off (Sincerely, Best regards, Cheers, Best, Warmly, Regards, Thank you)
        signoff_patterns = [
            r"\b(sincerely|best\s+regards|cheers|best|warmly|regards|thank\s+you|respectfully|kind\s+regards)\b"
        ]
        has_signoff = any(re.search(pat, email.strip(), re.IGNORECASE) for pat in signoff_patterns)

        # Calculate programmatic score (0.0 to 1.0)
        prog_score = 0.0
        prog_reasons = []
        if has_greeting:
            prog_score += 0.5
            prog_reasons.append("Greeting found")
        else:
            prog_reasons.append("No standard greeting found at start of email")

        if has_signoff:
            prog_score += 0.5
            prog_reasons.append("Sign-off found")
        else:
            prog_reasons.append("No standard sign-off found at end of email")

        # --- 2. LLM Checks (60% weight) ---
        prompt = (
            f"You are a communications editor. Review this email for correct paragraph layout and the "
            f"absence of conversational meta-talk (like 'Here is the email you requested:' or 'Let me know if you need anything else').\n\n"
            f"Generated Email:\n{email}\n\n"
            f"Evaluate and output the results matching the required output structure."
        )

        try:
            result: StructureResult = self.client.chat.completions.create(
                model=MODEL_A,
                response_model=StructureResult,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # Normalize 1-5 score to 0.0 - 1.0 range
            llm_layout_score = float((result.layout_score_1_to_5 - 1) / 4)
            # Penalize meta-text
            llm_score = llm_layout_score
            if result.has_meta_text:
                llm_score = max(llm_score - 0.5, 0.0)
                prog_reasons.append("Contains conversational LLM fluff/meta-text")
            
            # Combine scores
            self.score = (0.4 * prog_score) + (0.6 * llm_score)
            self.reason = (
                f"Combined score: {self.score:.2f} (Programmatic: {prog_score*100:.0f}%, "
                f"LLM: {llm_score*100:.0f}%). Details: {', '.join(prog_reasons)}. "
                f"LLM Analysis: {result.reasoning}"
            )
        except Exception as e:
            # Fallback to programmatic score only if LLM call fails
            self.score = prog_score
            self.reason = f"LLM check failed ({str(e)}). Programmatic fallback score: {prog_score:.2f}. Details: {', '.join(prog_reasons)}"

        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Structure and Professionalism Metric"


def evaluate_email_generation(
    intent: str,
    key_facts: List[str],
    tone: str,
    generated_email: str,
    reference_email: str
) -> Dict[str, Any]:
    """
    Orchestration wrapper to execute all three metrics on a single test case.
    """
    # Create DeepEval Test Case
    # Input is structured as a clear audit string containing intent and tone
    input_str = f"Intent: {intent} | Tone: {tone}"
    
    test_case = LLMTestCase(
        input=input_str,
        actual_output=generated_email,
        expected_output=reference_email,
        context=key_facts
    )

    # Instantiate custom metrics
    fact_metric = FactRecallMetric()
    tone_metric = ToneAdherenceMetric()
    struct_metric = StructurePolishMetric()

    # Run evaluations
    fact_score = fact_metric.measure(test_case)
    fact_reason = fact_metric.reason

    tone_score = tone_metric.measure(test_case)
    tone_reason = tone_metric.reason

    struct_score = struct_metric.measure(test_case)
    struct_reason = struct_metric.reason

    return {
        "fact_recall": {"score": fact_score, "reason": fact_reason},
        "tone_adherence": {"score": tone_score, "reason": tone_reason},
        "structure_polish": {"score": struct_score, "reason": struct_reason},
        "average_score": (fact_score + tone_score + struct_score) / 3.0
    }
