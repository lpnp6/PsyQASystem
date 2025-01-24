from extract import knowledge_extract
import json
from spark_ai import SparkPython
import asyncio
import traceback
import wikipedia
import spacy
import numpy as np

nlp = spacy.load('zh_core_web_sm')


def test_prompt():
    prompt = f"""帮我进行实体识别和关系提取,以下是实体类型:,以下是文本:。生成一个json文件，要求每个实体的模式为{{"type":TYPE,"name":NAME}}, type必须包含于给定的实体类型，每个关系的模式为{{"entity1":{{"type":type1,"name":name1}},"relation":RELATION,"entity2":{{"type":type2, "name":name2}}}},其中entity1和entity2必须包含于实体识别的结果。提取json文件格式要求{{"entities":[{{"type":type1,"name":name1}},{{"type":type2,"name":name2}}...],"relations":[{{"entity1":{{"type":type1, "name":name1}},"relation":RELATION,"entity2":{{"type":type2, "name":name2}}}}...]}}"""
    print(prompt)


async def test_database():
    handler = knowledge_extract.handler
    print(await handler.create_node("person", {"name": "lp"}))
    print(await handler.create_node("person", {"name": "lp"}))
    print(await handler.create_node("person", {"name": "pl"}))
    print(await handler.create_relationship("person",{"name": "lp"},"person", {"name": "pl"},knowledge_extract.clean_relationship("reversed name")))


async def test_filelist():
    count = 0
    async for file_path in knowledge_extract.list_file("data/nlpxinlixue"):
        count += 1
    print(count)
    
    
    
def test_wiki():
    print(wikipedia.summary("中国",sentences = 3))

test_wiki()