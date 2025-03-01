import rag.RAG as RAG
import configparser
import os
import asyncio
from rag.data import Document, Index
from rag.prompt import system_prompt
from rag.entity_relation_layer import update


config = configparser.ConfigParser()
config.read("d:/PsyQASystem/rag/api/config.ini", encoding="utf-8")

NEO4J_URI = config.get("NEO4J", "URI", fallback=None)
NEO4J_USERNAME = config.get("NEO4J", "USERNAME", fallback=None)
NEO4J_PASSWORD = config.get("NEO4J", "PASSWORD", fallback=None)
if NEO4J_URI:
    os.environ["NEO4J_URI"] = NEO4J_URI
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

SPARKAI_URL = config.get("SPARK", "URL", fallback=None)
SPARKAI_APP_ID = config.get("SPARK", "APP_ID", fallback=None)
SPARKAI_API_SECRET = config.get("SPARK", "API_SECRET", fallback=None)
SPARKAI_API_KEY = config.get("SPARK", "API_KEY", fallback=None)
SPARKAI_DOMAIN = config.get("SPARK", "DOMAIN", fallback=None)

if SPARKAI_URL:
    os.environ["SPARKAI_URL"] = SPARKAI_URL
    os.environ["SPARKAI_APP_ID"] = SPARKAI_APP_ID
    os.environ["SPARKAI_API_SECRET"] = SPARKAI_API_SECRET
    os.environ["SPARKAI_API_KEY"] = SPARKAI_API_KEY
    os.environ["SPARKAI_DOMAIN"] = SPARKAI_DOMAIN

DEEPSEEK_APIKEY = config.get("DEEPSEEK", "API_KEY", fallback=None)
os.environ["DEEPSEEK_APIKEY"] = DEEPSEEK_APIKEY

SILICONFLOW_APIKEY = config.get("SILICONFLOW", "API_KEY", fallback=None)
os.environ["SILICONFLOW_APIKEY"] = SILICONFLOW_APIKEY

ARK_APIKEY = config.get("ARK", "API_KEY", fallback=None)
os.environ["ARK_APIKEY"] = ARK_APIKEY

ELASTICSEARCH_URL = config.get("ELASTICSEARCH", "URL", fallback=None)
ELASTICSEARCH_APIKEY = config.get("ELASTICSEARCH", "API_KEY", fallback=None)
HIGH_LEVEL_INDEX_NAME = config.get(
    "ELASTICSEARCH", "HIGH_LEVEL_INDEX_NAME", fallback=None
)
LOW_LEVEL_INDEX_NAME = config.get(
    "ELASTICSEARCH", "LOW_LEVEL_INDEX_NAME", fallback=None
)
os.environ["ELASTICSEARCH_URL"] = ELASTICSEARCH_URL
os.environ["ELASTICSEARCH_APIKEY"] = ELASTICSEARCH_APIKEY
os.environ["HIGH_LEVEL_INDEX_NAME"] = HIGH_LEVEL_INDEX_NAME
os.environ["LOW_LEVEL_INDEX_NAME"] = LOW_LEVEL_INDEX_NAME

rag = RAG.RAG(llm_model_name="ArkDeepSeekChat")


async def test():
    text1 = """《永续成长的宝藏图》笔记\n模仿与策略的奥秘\n在《永续成长的宝藏图》中，乔瑟夫·欧可诺与伊安·麦克德摩深入探讨了模仿与策略在个人成长中的关键作用。书中提到，模仿不仅是学习的捷径，更是创新的源泉。例如，NLP（神经语言程序学）通过模仿成功人士的行为模式，帮助人们快速掌握复杂的技能，如领导力、沟通技巧等。书中还提到了一个具体的案例：伊安曾辅导一家汽车公司，通过模仿超级销售员的行为策略，显著提升了销售业绩。这些销售员首先通过视觉展示车辆，然后通过触觉让客户体验驾驶，最后再通过听觉讨论购车细节。这种策略的成功在于其多感官的整合，使得客户在购买决策中更加自信和满意。\n心理策略的深度解析\n心理策略是书中另一个重要的主题。作者指出，理解他人的心理策略是模仿的核心。心理策略涉及如何组织思想与行动以达成目标，从简单的记忆技巧到复杂的生涯规划。书中详细描述了如何通过分析表象系统、次感元和步骤顺序来模仿这些策略。例如，在拼字策略的案例中，作者展示了视觉表象系统在拼字中的重要性。一位学生在乔瑟夫的指导下，通过改变观想字体的背景颜色，显著提高了拼字能力。这个案例不仅展示了策略模仿的有效性，也强调了个性化学习的重要性。\n激励策略的多样性\n激励策略是推动个人成长的动力源泉。书中分析了两种截然不同的激励策略：一种是积极的，通过构建成功的心理画面来激励行动；另一种是消极的，通过想象失败的后果来迫使自己行动。例如，书中描述了一位模仿对象，她通过听到内部鼓励的声音和构建成功的心理画面来激励自己完成任务。相比之下，另一位模仿对象则通过想象不做事的负面后果来推动自己，这种策略虽然有效，但体验并不愉快。书中强调，选择合适的激励策略对于个人成长至关重要，它不仅影响行动的效率，还影响个人的心理健康。\n认知守门员的启示\n在书的最后部分，作者探讨了认知守门员——删减、扭曲和一般化——在塑造个人世界观中的作用。这些认知过程既是学习的工具，也是误解的源头。例如，删减可能导致我们忽略重要的反馈信息，而扭曲则可能让我们误解他人的意图。一般化虽然有助于快速反应，但也可能导致过度概括，阻碍新经验的接受。书中通过具体案例，如找不到钥匙后又在其原处找到的经历，展示了这些认知过程的实际影响。作者提醒读者，理解并管理这些认知守门员，是实现个人成长和创新的关键。\n\n通过这些深入的分析和具体的案例，《永续成长的宝藏图》不仅提供了理论框架，还为读者提供了实用的工具和策略，帮助他们在个人和职业生活中实现持续的成长和成功。"""
    document1 = Document(id="test1", token=len(text1), title="test1", text=text1)
    text2 = """《永续成长的宝藏图》笔记\n身心结合与脱离的艺术\n人类的心灵与身体，仿佛并蒂莲花，息息相关，共同构筑了我们的存在。当我们完全沉浸于当下，心心相印，这便是“结合”。曾几何时，在激烈争辩或热切讨论中，我们会不自觉地前倾，身体如同被磁力吸引，表现出深切的专注。反之，当我们陷入反思与检讨时，身体则往往向后靠，彷如离岸的帆船，远离风起云涌的现世片刻。这种“脱离”虽显疏离，却为我们提供了客观洞察的视角。\n结合与脱离两者并无优劣，关键在于灵活运用。能够自在游走于这两种状态之间，无疑是情绪清明与自我成长的奥秘所在。然而，不少人在过往的创伤中迷失，常陷于脱离的困境，思绪漂浮，心灵落寞。近年来，身心医学领域的研究表明，观想的力量对健康具有显著影响。例如🤖，一项2022年的研究发现，通过定期进行正念冥想，参与者的心理健康状况得到了显著改善，其焦虑程度降低了30%。\n模仿——渴望卓越的路径\n模仿，无疑是学习的捷径。基于神经语言程序(NLP)，模仿他人的卓越行为，便如同复制成功的秘密配方。小孩子观察成人学走路、学说话，所展示的正是这一非凡原理。美国著名幽默作家马克·吐温曾言，若是正式教导儿童学走路，他们也许不仅踉踉跄跄，还口吃不清。\n要有效模仿一项技能，需专注于以下三个层次：\n\n* 行为及生理语汇\n\n* 内心想法\n\n* 信念及价值观\n模仿的技巧不仅仅限于言语和动作，还包括更为复杂的心理层面。我们通常以观察、提问和跟随的方式进入模仿阶段，如同拼图般拆解每一个细节。若以销售策略为例，伊安曾辅导过一家汽车公司，发现他们的超级销售员在展示时先行视觉引导，再通过试驾带来触觉体验，最后才进入听觉层面讨论购车细节。如此多感官交流的策略，大大提升了销售成功率🚗。\n细分策略的力量\n模仿的过程犹如雕刻，需要逐步细分和打磨。NLP将这一过程系统化，通过细分目标并逐项攻克，使得模仿成为可能。以拼字为例，好的拼字者通常通过视觉系统进行记忆，而非听觉。乔瑟夫曾帮助一位拼字困难的学生调整观想背景颜色，短短两周，这位学生便一跃成为班级拼字高手之一。这种细腻的调整，体现了NLP在教育系统中的潜力。\n现实生活中，同样的理论应用广泛。谷歌公司针对员工的创新培训计划，便是通过模仿并细分成功项目，将复杂的创新过程分解为具体可行的步骤，使得每一位参与者都能逐步掌握并应用到自己的工作中。2020年，一项报告显示，经过谷歌培训的员工创新效率提升了25%✨。\n激励策略与焦虑容忍的对比\n激励与焦虑，仿佛两枚硬币的两面，前者点燃我们前进的火焰，后者则如山雨欲来，使我们踟蹰不前。有效的激励策略通常将未来的成功形象化，通过积极的内部对话，赋予我们动力。例如，一位优秀的执行者会首先设想任务完成后的光景，内心中听到强有力的鼓励声音，然后全情投入。而反面的策略则是看到未完成任务的后果，听到尖酸刻薄的自我批评声音，结果只是不断地拖延。\n一个典型的案例便是高斯大学的一份研究，发现使用建设性激励策略的学生，其年度综合表现较未使用者高出20%。反之，那些充满自我批评和负面预测的学生，则容易陷入焦虑之中，导致表现不佳🧠。\n在心理学研究领域，焦虑策略的分析尤为重要。焦虑往往源自对未来不确定性的恐惧，通过构建失败的生动情景，焦虑的心灵仿佛锁链重生，使人无法迈步向前。然而，正如NLP所言，人类本非错漏之物，通过揭示运作的机制，助其实现自我变化，乃是通向卓越的关键。\n综上所述，《永续成长的宝藏图》如同一部心灵指南，通过结合与脱离的平衡、模仿成功的策略、系统化的细分技巧，以及有效的激励与焦虑管理，助力我们在生活的多舛航程中找到那一份内心的平静与卓越的可能。这无疑是现代人心灵成长与自我提升的珍贵宝典。"""
    document2 = Document(id="test2", token=len(text2), title="test2", text=text2)
    chunks1 = await rag.insert_document(document1)
    chunks2 = await rag.insert_document(document2)
    tasks = []
    tasks.append(asyncio.create_task(rag.build_kg(chunks1)))
    tasks.append(asyncio.create_task(rag.build_kg(chunks2)))
    results = await asyncio.gather(*tasks)
    for result in results:
        await update(result, rag.vector_db_handler, rag.graph_db_handler)
    return results


asyncio.run(test())
