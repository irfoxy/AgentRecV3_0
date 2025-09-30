from ddgs import DDGS
from langchain_core.tools import tool
import re
import json

MAX_SEARCH_RESULTS=10

@tool
def search(key_words: str):
    """
    Use DuckDuckGo search engine (DDGS) to perform a text search based on the given keywords.

    Parameters:
    - key_words (str): The keywords or phrase used for the search.

    Returns:
    - list: A list containing the text content of the search results, with a maximum of 10 results.
    """
    return DDGS().text(query=key_words, max_results=MAX_SEARCH_RESULTS)

def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("JSON NOT FOUND")
    
    json_str = match.group(0).strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON ERROR: {e}")