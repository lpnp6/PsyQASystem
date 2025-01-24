import json
import asyncio
from spark_ai import SparkPython
from pathlib import Path
from knowledge_graph import graph_database_handler
import re
import traceback
import wikipedia
import spacy
import numpy as np
from hanziconv import HanziConv

convert = HanziConv()
nlp = spacy.load("zh_core_web_sm")
wikipedia.set_lang("zh")

with open("config/neo4j_config.json", "r") as f:
    config = json.load(f)
    handler = graph_database_handler.GraphDatabaseHandler(
        url=config["NEO4J_URL"],
        user=config["NEO4J_USERNAME"],
        password=config["NEO4J_PASSWORD"],
    )


# "NEO4J_URL": "neo4j+s: //798d60cd.databases.neo4j.io"
# "NEO4J_USERNAME": "neo4j",
# "NEO4J_PASSWORD": "fVE6Q1OniFxa9CBUANV8_ShH42Y8ooi-0TpFZ5r-U4Q"
# "AURA_INSTANCEID": "798d60cd"
# "AURA_INSTANCENAME": "Instance01"

with open("extract/simplified_schema.json", "r", encoding="utf-8") as f:
    simplified_schema = json.load(f)["entities"]


async def ner(text):
    prompt = f"""帮我进行实体识别,以下是文本:{text}。生成一个json文件，每个实体的模式为(type,name)。json文件格式要求[{{"type":type1,"name":name1}},{{"type":type2,"name":name2}}...]"""
    repsonse = SparkPython.query(prompt)
    match = re.search(r"```json(.+)```", repsonse, re.DOTALL)
    if match:
        return json.loads(match.group(1).lstrip("```json").rstrip("```"))
    else:
        return []


async def extract(text):
    prompt = f"""帮我进行实体识别和关系提取,以下是实体类型{simplified_schema},以下是文本:{text}。生成一个json文件，要求有且只有一个json代码块，每个实体的模式为{{"type":TYPE,"name":NAME}}, type必须包含于给定的实体类型，每个关系的模式为{{"entity1":{{"type":type1,"name":name1}},"relation":RELATION,"entity2":{{"type":type2, "name":name2}}}},其中entity1和entity2必须包含于实体识别的结果。提取json文件格式要求{{"entities":[{{"type":type1,"name":name1}},{{"type":type2,"name":name2}}...],"relations":[{{"entity1":{{"type":type1, "name":name1}},"relation":RELATION,"entity2":{{"type":type2, "name":name2}}}}...]}}"""
    repsonse = SparkPython.query(prompt)
    match = re.search(r"```json(.+)```", repsonse, re.DOTALL)
    if match:
        return json.loads(match.group(1).lstrip("```json").rstrip("```"))
    else:
        return []


async def list_file(dir):
    with open("extract\\success.log", "r", encoding="utf-8") as f:
        text = f.read()
        matches = re.findall(r"nlpxinlixue\d+\.txt", text)
    path = Path(dir)
    for f in path.rglob("*"):
        if f.is_file():
            if f.name not in matches:
                yield f


def clean_relationship(relationship: str):
    return re.sub(
        r"[^\w\s]",
        "",
        relationship.replace("...", "")
        .replace("，", "")
        .replace("…", "")
        .replace(" ", ""),
    )


async def schema():
    schema = {"entities": []}
    count = 0
    async for file_path in list_file("data/nlpxinlixue"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            result = await ner(text)
            for entity in result:
                type = entity["type"].replace(" ", "_")
                name = entity["name"]
                if type not in schema["entities"]:
                    schema["entities"].append(type)
                count += 1
                print(str(count) + "\n")
        except Exception as e:
            try:
                print(e)
                with open("spark_ai/spark_ai.log", "a", encoding="utf-8") as f:
                    f.write(str(file_path) + "\n")
                    f.write(traceback.format_exc() + "\n")
                    f.write(str(SparkPython.text) + "\n")
            except Exception as e:
                print(e)
    with open("extract/schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=4)


async def build():
    success_log = open("extract/success.log", "a", encoding="utf-8")
    error_log = open("extract/error.log", "a", encoding="utf-8")
    async for file_path in list_file("data/nlpxinlixue"):
        try:
            error_log.write(str(file_path) + "\n")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                result = await extract(text)
            except Exception as e:
                error_log.write(traceback.format_exc() + "\n")
                error_log.write(str(SparkPython.text) + "\n")
                continue
            entities = result["entities"]
            relations = result["relations"]
            for entity in entities:
                type = entity["type"].replace(" ", "_")
                name = entity["name"]
                if type not in simplified_schema:
                    simplified_schema["entities"].append(type)
                try:
                    await handler.create_node(
                        type,
                        {
                            "name": name,
                        },
                    )
                except Exception as e:
                    error_log.write(traceback.format_exc() + "\n")
                    error_log.write(traceback.format_exc() + "\n")
                    continue
            for relation in relations:
                entity1 = relation["entity1"]
                entity2 = relation["entity2"]
                relationship = relation["relation"]
                type1 = entity1["type"]
                type2 = entity2["type"]
                name1 = entity1["name"]
                name2 = entity2["name"]
                try:
                    await handler.create_relationship(
                        type1, {"name": name1}, type2, {"name": name2}, relationship
                    )
                except Exception as e:
                    error_log.write(traceback.format_exc() + "\n")
                    error_log.write(traceback.format_exc() + "\n")
                    continue
            success_log.write(str(file_path) + "\n")
            success_log.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            try:
                with open("spark_ai/spark_ai_error.log", "a", encoding="utf-8") as f:
                    f.write(str(file_path) + "\n")
                    f.write(traceback.format_exc() + "\n")
                    f.write(str(SparkPython.text) + "\n")
            except Exception as e:
                print(e)
    success_log.close()
    error_log.close()


if __name__ == "__main__":
    asyncio.run(build())
