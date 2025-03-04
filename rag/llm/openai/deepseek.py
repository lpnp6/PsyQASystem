from rag.llm.openai.openai_base import OpenaiBase
from dataclasses import dataclass
import os
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
class ArkDeepSeekChat(OpenaiBase):
    def __init__(self, model_name, config=None):
        super().__init__(
            model_name="deepseek-v3-241226",
            config=config,
            base_url="https://ark.cn-beijing.volces.com/api/v3/",
            api_key=os.environ["ARK_APIKEY"],
        )


@dataclass
class OllamaDeepSeekR1(OpenaiBase):
    def __init__(self, model_name=None, config=None):
        super().__init__(
            model_name="deepseek-r1",
            config=config,
            base_url="https://10.29.0.1:8000/",
            api_key="ollama",
        )


llm = OllamaDeepSeekR1(config=dict())
print(asyncio.run(llm.generate("字节跳动位于北京")))
