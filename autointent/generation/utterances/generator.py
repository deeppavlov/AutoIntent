import os

import openai
from dotenv import load_dotenv


class Generator:
    def __init__(self):
        load_dotenv()
        self.client = openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = os.environ["OPENAI_MODEL_NAME"]

    def get_chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content
