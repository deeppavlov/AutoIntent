import openai
from dotenv import load_dotenv
import os


class Generator:
    def __init__(self):
        load_dotenv()
        self.client = openai.OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
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

# def get_chat_completion(sampling_params: dict[str, Any]):
#     if "temperature" not in sampling_params:
#         sampling_params["temperature"] = 1
#     if "top_p" not in sampling_params:
#         sampling_params["top_p"] = 1
#     if "max_tokens" not in sampling_params:
#         sampling_params["max_tokens"] = 3000
#     if "stop" not in sampling_params:
#         sampling_params["stop"] = []
    
    
#     client.chat.completions.create(
#         model=self.model,
#         messages=messages,
#         temperature=sampling_params["temperature"],
#         top_p=sampling_params["top_p"],
#         stop=sampling_params["stop"],
#         max_tokens=sampling_params["max_tokens"],
#         extra_body={"min_p": sampling_params["min_p"]},
#     )
