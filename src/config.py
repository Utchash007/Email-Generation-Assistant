import os
from dotenv import load_dotenv
import openai
import instructor

# Load environment variables from .env file
load_dotenv()

# Setup model name constants
MODEL_A = "llama-3.1-8b-instant"
MODEL_B = "openai/gpt-oss-120b"

# Setup Cerebras connection details
CEREBRAS_API_BASE = "https://api.groq.com/openai/v1"

def get_instructor_client(mode: instructor.Mode = instructor.Mode.TOOLS) -> instructor.Instructor:
    """
    Initializes and returns an Instructor patched OpenAI client for Cerebras Inference with a specific mode.
    """
    # Read Cerebras API key, falling back to OpenRouter key name for ease of use
    api_key = os.environ.get("CEREBRAS_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Neither CEREBRAS_API_KEY nor GROQ_API_KEY is set in your .env file.")
    
    # Initialize standard OpenAI client pointed at Cerebras base URL
    raw_client = openai.OpenAI(
        base_url=CEREBRAS_API_BASE,
        api_key=api_key,
    )
    
    # Patch client with Instructor for structured output validation using target mode
    return instructor.from_openai(raw_client, mode=mode)
