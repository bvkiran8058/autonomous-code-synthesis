import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# Load environment variables (useful for local testing)
load_dotenv()

# Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_RETRIES = 3
TOP_K_RESULTS = 5
EXECUTION_TIMEOUT = 15  # seconds
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Initialize and return a ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(
        model=model_name, 
        api_key=GEMINI_API_KEY, 
        temperature=temperature,
        max_retries=MAX_RETRIES
        )

