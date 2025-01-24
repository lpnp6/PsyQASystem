# coding: utf-8
from spark_ai import SparkApi
import json

with open("config/spark_max.json", "r") as f:
    config = json.load(f)

# "SPARKAI_URL": "wss://spark-api.xf-yun.com/v1.1/chat",
# "SPARKAI_APP_ID": "55708d47",
# "SPARKAI_API_SECRET": "OGU1OTMxZTM4ZTY4MDE2M2M1NDFmNjk1",
# "SPARKAI_API_KEY": "776a893b9534f3eda5d5cca6c4646ee8",
# "SPARKAI_DOMAIN": "lite"

api_url = config["SPARKAI_URL"]
appid = config["SPARKAI_APP_ID"]
api_secret = config["SPARKAI_API_SECRET"]
api_key = config["SPARKAI_API_KEY"]
domain = config["SPARKAI_DOMAIN"]

# 初始上下文内容，当前可传system、user、assistant 等角色
text = [
    # {"role": "system", "content": "你现在扮演李白，你豪情万丈，狂放不羁；接下来请用李白的口吻和用户对话。"} , # 设置对话背景或者模型角色
    # {"role": "user", "content": "你是谁"},  # 用户的历史问题
    # {"role": "assistant", "content": "....."} , # AI的历史回答结果
    # # ....... 省略的历史对话
    # {"role": "user", "content": "你会做什么"}  # 最新的一条问题，如无需上下文，可只传最新一条问题
]


def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while getlength(text) > 8000:
        del text[0]
    return text


def query(text):
    question = checklen(getText("user", text))
    SparkApi.answer = ""
    SparkApi.main(appid, api_key, api_secret, api_url, domain, question)
    getText("assistant", SparkApi.answer)
    return SparkApi.answer

