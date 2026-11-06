from pydantic import BaseModel

from src.hospital_rag.interfaces import Embedder, LLMClient, SearchHit, VectorStore

REFUSAL = "I don't know based on the available SOPs."


class Answer(BaseModel):
    text: str
    sources: list[SearchHit]
    refused: bool


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}]\n{hit.chunk.text}")
    context = "\n\n".join(blocks)
    return (
        "You are an assistant for hospital staff. Answer the question using "
        "ONLY the context below. Cite sources only by number, like [1] or [2]. "
        "Do not write SOP ids, page numbers or versions yourself. "
        "If the context does not contain the answer, reply exactly: "
        f'"{REFUSAL}"\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_question(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    top_k: int = 3,
    min_score: float = 0.15,
) -> Answer:
    hits = store.search(embedder.embed([question])[0], top_k=top_k)
    if not hits or hits[0].score < min_score:
        return Answer(text=REFUSAL, sources=[], refused=True)
    text = llm.generate(build_prompt(question, hits)).strip()
    return Answer(text=text, sources=hits, refused=REFUSAL in text)