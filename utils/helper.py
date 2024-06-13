import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())

def get_open_ai_model_and_key():   
    load_env()
    key = os.getenv("OPEN_API_KEY")
    model = os.getenv("MODEL")
    return key, model

if __name__ == "__main__":
    print(get_open_ai_model_and_key())