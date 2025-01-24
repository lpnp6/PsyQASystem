from neo4j import AsyncGraphDatabase


class GraphDatabaseHandler:
    def __init__(self, url, user, password):
        self.driver = AsyncGraphDatabase.driver(url, auth=(user, password))

    def close(self):
        self.driver.close()

    # 创建节点
    async def create_node(self, label, properties):
        query = f"""
        MERGE (n:{label} {{name: $name}})
        on CREATE SET n = $properties
        return n
        """
        async with self.driver.session() as session:
            result = await session.run(
                query, name=properties["name"], properties=properties
            )
            return await result.single()

    # 创建关系
    async def create_relationship(
        self, label1, property1, label2, property2, relationship
    ):
        query = (
            f"MERGE (a:{label1} {{name: $name1}})"
            f"MERGE(b:{label2} {{name: $name2}})"
            f"MERGE (a)-[r:{relationship}]->(b)"
            f"RETURN r"
        )
        async with self.driver.session() as session:
            result = await session.run(
                query,
                name1=property1["name"],
                name2=property2["name"],
            )
            return await result.single()

    # 查询所有节点
    async def get_nodes(self):
        query = "MATCH (n) RETURN n"
        with self.driver.session() as session:
            result = await session.run(query)
            return [record["n"] for record in result]
