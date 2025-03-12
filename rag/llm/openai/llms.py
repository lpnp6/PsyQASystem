from rag.llm.openai.openai_base import OpenaiBase
from dataclasses import dataclass
import os
import openai
import asyncio


@dataclass
class DeepSeekChat(OpenaiBase):
    def __init__(self, model_name="deepseek-chat", config=None):
        super().__init__(
            model_name="deepseek-chat",
            config=config,
            base_url="https://api.deepseek.com/beta",
            api_key=os.environ["DEEPSEEK_APIKEY"],
        )


@dataclass
class DeepSeekR1(OpenaiBase):
    def __init__(self, model_name="deepseek-reasoner", config=None):
        super().__init__(
            model_name="deepseek-reasoner",
            config=config,
            base_url="https://api.deepseek.com/beta",
            api_key=os.environ["DEEPSEEK_APIKEY"],
        )


@dataclass
class SiliconFlowDeepSeekChat(OpenaiBase):
    def __init__(self, model_name, config=None):
        super().__init__(
            model_name="deepseek-ai/DeepSeek-V2.5",
            config=config,
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.environ["SILICONFLOW_APIKEY"],
        )


@dataclass
class SiliconFlowDeepSeekR1(OpenaiBase):
    def __init__(self, model_name, config=None):
        super().__init__(
            model_name="deepseek-ai/DeepSeek-R1",
            config=config,
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.environ["SILICONFLOW_APIKEY"],
        )


@dataclass
class Ark(OpenaiBase):
    def __init__(self, model_name, config=None):
        super().__init__(
            model_name="doubao-1-5-pro-32k-250115",
            config=config,
            base_url="https://ark.cn-beijing.volces.com/api/v3/",
            api_key=os.environ["ARK_APIKEY"],
        )





