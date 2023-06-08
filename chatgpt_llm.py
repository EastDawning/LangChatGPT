import json
from langchain.llms.base import LLM
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from configs.model_config import *
from utils import torch_gc
import openai


class ChatGPT(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self,api_key):
        super().__init__()
        openai.api_key = api_key
        self.chatgpt=openai

    @property
    def _llm_type(self) -> str:
        return "ChatGPT"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:
        response=self.chatgpt.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\"\"\""]
        ).choices[0].message.content

        history += [[prompt, response]]
        yield response, history


    def load_model(self,
                   model_name_or_path: str =None,
                   llm_device=LLM_DEVICE,
                   use_ptuning_v2=False,
                   use_lora=False,
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        print("openai 模型加载成功")


if __name__ == "__main__":
    llm = ChatGPT()
    llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL],
                   llm_device=LLM_DEVICE, )
    last_print_len = 0
    for resp, history in llm._call("你好", streaming=True):
        print(resp[last_print_len:], end="", flush=True)
        last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        print(resp)
    pass
