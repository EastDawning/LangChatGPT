import json
from langchain.llms.base import LLM
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from configs.model_config import *
from utils import torch_gc
import openai


class ChatGPT(LLM):
    max_token: int = 2000
    temperature: float = 0.01
    top_p = 0.9
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self, api_key):
        super().__init__()
        openai.api_key = api_key

    @property
    def _llm_type(self) -> str:
        return "ChatGPT"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{"role": ROLE, "content": prompt}],
                                                temperature=TEMPERATURE,
                                                max_tokens=MAX_TOKEN,
                                                top_p=TOP_P,
                                                frequency_penalty=FREQUENCECY_PENALTY,
                                                presence_penalty=PRESENCE_PENALTY,
                                                stop=STOP
                                                ).choices[0].message.content
        history += [[prompt, response]]
        yield response, history

    def load_model(self,
                   model_name_or_path: str = None,
                   llm_device=LLM_DEVICE,
                   use_ptuning_v2=False,
                   use_lora=False,
                   device_map: Optional[Dict[str, int]] = None,
                   **kwargs):
        raise NameError
        print("openai 模型加载失败.")


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
