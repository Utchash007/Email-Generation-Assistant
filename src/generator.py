import instructor
from src.models import EmailScenario, EmailGenerationResponse
from src.config import get_instructor_client

class EmailGenerator:
    def __init__(self):
        self.client = get_instructor_client(mode=instructor.Mode.MD_JSON)


    def generate_email(self, scenario: EmailScenario, model_name: str) -> EmailGenerationResponse:
        """
        Generates a structured email response for a given scenario and model using a layered prompting strategy.
        """
        system_instruction = (
            "You are an expert Corporate and Personal Communications Specialist. Your goal is to write high-quality, "
            "effective emails based on a given Intent, a list of Key Facts, and a desired Tone.\n\n"
            "You MUST apply the following prompt engineering techniques:\n"
            "1. Role-Playing: Adopt the persona of a highly skilled communications manager. Match the language, vocabulary, "
            "and style to the target Tone (formal, casual, urgent, empathetic).\n"
            "2. Chain-of-Thought: Before writing the subject line and body, plan your approach step-by-step in the 'thinking' "
            "field. List each key fact and describe how it will be mapped into the email text, and explain how you will establish the requested tone.\n"
            "3. Few-Shot Examples: Study the examples below to understand the expected format and quality standards. Do not output any "
            "chat dialog or wrapper text outside of the structured fields.\n\n"
            "--- FEW-SHOT EXAMPLE 1 ---\n"
            "Input Scenario:\n"
            "Intent: Request updates on the bug fix from the engineering team.\n"
            "Key Facts:\n"
            "- Core platform crash occurring on login screen for iOS users.\n"
            "- Critical hotfix was promised by Thursday noon.\n"
            "- Client is demanding status update.\n"
            "Tone: urgent\n"
            "Expected Output Structure:\n"
            "thinking: [The tone is urgent because a crash is affecting users and a client is asking for status. I will plan to address the lead engineer, list the login crash fact, reference the Thursday noon deadline, and ask for an immediate status update without filler.]\n"
            "email_subject: URGENT: Status Update Needed - iOS Login Crash Hotfix\n"
            "email_body: Hi Team,\n\nWe need an immediate status update on the iOS login screen crash hotfix. As discussed, this crash is affecting all core platform users on iOS, and the client is currently demanding a progress report. The hotfix was scheduled to be completed by Thursday noon. Please let me know the current status and expected release time as soon as possible.\n\nBest regards,\n[Name]\n\n"
            "--- FEW-SHOT EXAMPLE 2 ---\n"
            "Input Scenario:\n"
            "Intent: Check in on a colleague after they returned from sick leave.\n"
            "Key Facts:\n"
            "- Out of office for 3 days due to flu.\n"
            "- Want to offer help catch up on email backlog.\n"
            "- Suggest coffee break if they feel up to it.\n"
            "Tone: casual\n"
            "Expected Output Structure:\n"
            "thinking: [The tone is casual and empathetic. I will check in friendly, mention their 3 days off with flu, offer help catch up on work/backlog, and suggest a coffee chat if they are feeling up to it.]\n"
            "email_subject: Welcome back! / Catching up\n"
            "email_body: Hi David,\n\nWelcome back to the office! I hope you're feeling much better after being out with the flu this week.\n\nDon't stress about catching up on everything at once. If you need any help sorting through your email backlog or getting up to speed on projects, let me know. When you get a chance and feel up to it, let's grab a coffee to catch up.\n\nCheers,\n[Name]"
        )

        user_content = (
            f"Input Scenario to Generate:\n"
            f"Intent: {scenario.intent}\n"
            f"Key Facts:\n"
            + "\n".join(f"- {fact}" for fact in scenario.key_facts)
            + f"\nTone: {scenario.tone}"
        )

        # Call OpenRouter API with Pydantic validation
        response: EmailGenerationResponse = self.client.chat.completions.create(
            model=model_name,
            response_model=EmailGenerationResponse,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3, # Lower temperature for consistency and adherence
        )

        return response
