if __name__ == "__main__":
    from rag.embedding import Spacy
    from rag.retrieve_layer import extract_subject_predicate
    import asyncio

    print(asyncio.run(Spacy().get_embeddings("伊安·麦克德摩")))