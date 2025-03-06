from rag.base import BaseLLM
from dataclasses import dataclass
from openai import OpenAI
from asyncio import Lock
import os
import time


@dataclass
class OpenaiBase(BaseLLM):
    def __init__(self, model_name, config, base_url, api_key):
        super().__init__(model_name=model_name, config=config)
        self.base_url = base_url
        self.system_prompt = config.get("system_prompt")
        self.response_prefix = config.get("response_prefix")
        self.json_format = config.get("json_format")
        self.__api_key = api_key
        self.client = OpenAI(api_key=self.__api_key, base_url=self.base_url)
        self.context = []
        self.context_lock = Lock()
        if self.system_prompt is not None:
            self.context.append({"role": "system", "content": self.system_prompt})

    def change_system_prompt(self, system_prompt=None):
        self.context.clear()
        self.system_prompt = system_prompt
        if self.system_prompt is not None:
            self.context.append({"role": "system", "content": self.system_prompt})
        if self.response_prefix is not None:
            self.context.append(
                {"role": "assistant", "content": self.response_prefix, "prefix": True}
            )

    async def generate(self, prompt, max_retries=3, **kwargs):
        def get_len(context):
            length = 0
            for content in context:
                length += len(content["content"])
            return length

        def check_len(context):
            max_length = 8000
            while get_len(context) > max_length:
                if context[0]["role"] == "system":
                    if len(context) >= 3:
                        del context[1:3]
                    else:
                        break
                else:
                    if len(context) >= 2:
                        del context[0:2]
                    else:
                        break
            return context

        def get_context(role, content, prefix=False):
            jsoncon = {}
            jsoncon["role"] = role
            jsoncon["content"] = content
            if prefix:
                jsoncon["prefix"] = True
            self.context.append(jsoncon)
            return self.context

        async with self.context_lock:
            messages = check_len(get_context("user", prompt))
            if self.response_prefix is not None:
                messages = check_len(
                    get_context("assistant", self.response_prefix, prefix=True)
                )
        for attempt in range(max_retries):
            try:
                if self.json_format:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.2
                    )
                else:
                     result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2
                    )
                response = result.choices[0].message.content
                async with self.context_lock:
                    get_context("assistant", response)
                return response
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return e
