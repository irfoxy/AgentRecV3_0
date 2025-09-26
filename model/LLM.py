import getpass
import os

os.environ["OPENAI_API_KEY"] = "HLyIME7oELKA3TgDN4kdgHpHNChoHxi5LwlyoObMfppukGKw"
os.environ["OPENAI_BASE_URL"] = "https://sg.uiuiapi.com/v1"

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

GPT = init_chat_model("gpt-4.1", model_provider="openai")