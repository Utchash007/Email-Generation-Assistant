from pydantic import BaseModel, Field
from typing import List

class EmailScenario(BaseModel):
    id: int
    intent: str = Field(description="The core purpose or intent of the email.")
    key_facts: List[str] = Field(description="Bullet points of key facts that must be included in the email.")
    tone: str = Field(description="The desired tone style (formal, casual, urgent, empathetic).")
    reference_email: str = Field(description="Human-written reference email for evaluation comparison.")

class EmailGenerationResponse(BaseModel):
    thinking: str = Field(
        description="Step-by-step thinking outlining how the intent and tone are addressed, and planning how all key facts will be seamlessly incorporated into the email before drafting it."
    )
    email_subject: str = Field(description="The subject line of the email.")
    email_body: str = Field(
        description="The final professional email text containing greetings, body paragraphs, and a professional sign-off. Do not include any extra chat dialogue before or after the email."
    )
