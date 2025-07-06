import os

def load_env():
    # 간단한 환경 변수 로딩
    pass

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        openai_api_key = input("🔑Insert your OpenAI API key: ").strip()

    return openai_api_key