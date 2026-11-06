from src.hospital_rag.fakes import FakeEmbedder, InMemoryVectorStore
from src.hospital_rag.interfaces import Chunk
from src.hospital_rag.rag import REFUSAL, answer_question


class RecordingLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def make_store(embedder: FakeEmbedder) -> InMemoryVectorStore:
    chunk = Chunk(
        id="c1",
        document_id="doc1",
        text="Apply hand rub for 20 to 30 seconds.",
        metadata={
            "sop_id": "SOP-X",
            "section": "Procedure",
            "page": 2,
            "version": "2.1",
        },
    )
    store = InMemoryVectorStore()
    store.add([chunk], embedder.embed([chunk.text]))
    return store


def test_empty_store_refuses_without_calling_llm():
    llm = RecordingLLM("should not be used")

    result = answer_question(
        "anything", FakeEmbedder(), InMemoryVectorStore(), llm
    )

    assert result.refused
    assert result.text == REFUSAL
    assert llm.prompts == []


def test_low_score_refuses_without_calling_llm():
    embedder = FakeEmbedder()
    llm = RecordingLLM("should not be used")

    result = answer_question(
        "unrelated question", embedder, make_store(embedder), llm, min_score=1.5
    )

    assert result.refused
    assert llm.prompts == []


def test_prompt_contains_numbered_context_but_no_page_or_version():
    embedder = FakeEmbedder()
    llm = RecordingLLM("Apply for 20 to 30 seconds [1].")

    result = answer_question(
        "Apply hand rub for 20 to 30 seconds.", embedder, make_store(embedder), llm
    )

    prompt = llm.prompts[0]
    assert "[1]\nApply hand rub for 20 to 30 seconds." in prompt
    assert "page 2" not in prompt
    assert "version 2.1" not in prompt
    assert not result.refused
    assert result.sources[0].chunk.metadata["page"] == 2


def test_model_refusal_is_detected():
    embedder = FakeEmbedder()
    llm = RecordingLLM(REFUSAL)

    result = answer_question(
        "Apply hand rub for 20 to 30 seconds.", embedder, make_store(embedder), llm
    )

    assert result.refused


def test_prompt_tells_model_not_to_write_citation_details():
    embedder = FakeEmbedder()
    llm = RecordingLLM("ok")

    answer_question(
        "Apply hand rub for 20 to 30 seconds.", embedder, make_store(embedder), llm
    )

    assert "Do not write SOP ids, page numbers or versions" in llm.prompts[0]