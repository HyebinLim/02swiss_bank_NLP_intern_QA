import os
from dotenv import load_dotenv, find_dotenv
                                                                                                                                  
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        openai_api_key = input("🔑Insert your OpenAI API key: ").strip()

    return openai_api_key