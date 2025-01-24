import httpx
from parsel import Selector

BASE_HOST = "https://www.nlpxinlixue.cn"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


class article:
    def __init__(self, title, link, content):
        self.title = title
        self.link = link
        self.content = content

    def __str__(self):
        return self.title + "\n" + self.content

    def __repr__(self):
        return self.title + "\n" + self.content


async def get_catalog(page_num):
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(
            BASE_HOST + "/page/" + str(page_num), headers=headers
        )
        if response.status_code != 200:
            return Exception(response.text)
        selector = Selector(text=response.text)
        catalog = []
        for article_panel in selector.css("div.article-panel"):
            title = article_panel.css("h3 a::text").get()
            link = article_panel.css("h3 a::attr(href)").get()
            catalog.append(article(title, link, ""))
        return catalog


async def get_article_content(link):
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(link, headers=headers)
        if response.status_code != 200:
            return Exception(response.text)
        selector = Selector(text=response.text)
        return "\n".join(selector.css("div.content").css("*::text").getall()).strip()


async def crawler():
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.get(BASE_HOST, headers=headers)
        if response.status_code != 200:
            return Exception(response.text)
        selector = Selector(text=response.text)
        page_num = int(selector.css("div.paginations a::text").getall()[-1])
        print("Total pages:", page_num)
    all_articles = []
    index = 1
    for i in range(1, page_num + 1):
        catalog = await get_catalog(i)
        for article in catalog:
            article.content = await get_article_content(article.link)
            with open(
                f"data/nlpxinlixue/nlpxinlixue{index}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(str(article) + "\n\n")
            index += 1


if __name__ == "__main__":
    import asyncio

    asyncio.run(crawler())
