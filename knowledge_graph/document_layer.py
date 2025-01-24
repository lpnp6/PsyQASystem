import spacy
import wikipedia
from hanziconv import HanziConv
import httpx
from parsel import Selector

convert = HanziConv()

async def get_wiki_article_text(entity_name):
    url = "https://zh.wikipedia.org/wiki/{entity_name}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if(response.status_code != 200):
            raise Exception(response.text)
        selector = Selector(response.text)
        css = selector.css("div.mw-body-content")
        if(css.css())

async def add_document_to_db(entity_name):
    pass