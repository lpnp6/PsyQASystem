from rag.base import BaseLLM
from dataclasses import dataclass
from rag.llm.spark import SparkApi
from asyncio import Lock
import os


@dataclass
class Spark(BaseLLM):
    def __init__(self, model_name, config):
        super().__init__(model_name=model_name, config=config)
        self.sparkai_url = os.environ.get("SPARKAI_URL")
        self.sparkai_app_id = os.environ.get("SPARKAI_APP_ID")
        self.__sparkai_api_secret = os.environ.get("SPARKAI_API_SECRET")
        self.__sparkai_api_key = os.environ.get("SPARKAI_API_KEY")
        self.sparkai_domain = os.environ.get("SPARKAI_DOMAIN")
        self.system_prompt = config["system_prompt"]  # 默认值 None
        self.context = []
        self.context_lock = Lock()
        if self.system_prompt is not None:
            self.context.append({"role": "system", "content": self.system_prompt})

    async def change_system_prompt(self, system_prompt=None):
        self.context.clear()
        self.system_prompt = system_prompt
        if self.system_prompt is not None:
            self.context.append({"role": "system", "content": self.system_prompt})

    async def generate(self, prompt, **kwargs):
        def get_len(context):
            length = 0
            for content in context:
                length += len(content["content"])
            return length

        def check_len(context):
            if get_len(context) > 8000:
                if self.system_prompt is not None:
                    del context[1]  # 删除第一轮对话的user_prompt和response
                    del context[2]
                else:
                    del context[0]
                    del context[1]
            return context

        def get_context(role, content):
            self.context.append({"role": role, "content": content})
            return self.context

        async with self.context_lock:
            messages = check_len(get_context("user", prompt))
        SparkApi.answer = ""
        SparkApi.main(
            self.sparkai_app_id,
            self.__sparkai_api_key,
            self.__sparkai_api_secret,
            self.sparkai_url,
            self.sparkai_domain,
            messages,
        )
        async with self.context_lock:
            get_context("assistant", SparkApi.answer)
        return SparkApi.answer
