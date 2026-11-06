from src.hospital_rag.config import get_settings
from src.hospital_rag.embedder import SentenceTransformerEmbedder
from src.hospital_rag.milvus_store import MilvusVectorStore

ANSWERABLE = [
    ("What ID proof is needed to admit a patient?", "SOP-ADM-001"),
    ("When should wristbands be fitted on admitted patients?", "SOP-ADM-001"),
    ("How long should alcohol hand rub be applied?", "SOP-INF-002"),
    ("When must I clean my hands around a patient?", "SOP-INF-002"),
    ("What are the five rights of medication?", "SOP-PHR-003"),
    ("What must a nurse do after receiving a verbal order?", "SOP-PHR-003"),
    ("What is needed in a discharge summary?", "SOP-DIS-004"),
    ("What if a patient wants to leave against medical advice?", "SOP-DIS-004"),
    ("Which number do I call for a cardiac arrest?", "SOP-EMG-005"),
    ("What is checked on the crash cart?", "SOP-EMG-005"),
    ("What colour bag is used for different waste?", "SOP-BMW-006"),
    ("How long can waste be stored before disposal?", "SOP-BMW-006"),
    ("In what order should blood tubes be drawn?", "SOP-LAB-007"),
    ("When are lab samples rejected?", "SOP-LAB-007"),
]

UNANSWERABLE = [
    "What is the visiting policy for the cafeteria?",
    "How do I apply for annual leave?",
    "What is the parking fee for visitors?",
    "Who won the cricket match yesterday?",
    "What are the ICU visiting hours?",
]

settings = get_settings()
embedder = SentenceTransformerEmbedder()
store = MilvusVectorStore(
    dimension=embedder.dimension,
    collection_name=settings.milvus_collection,
    uri=settings.milvus_uri,
)


def top_hits(question: str):
    return store.search(embedder.embed([question])[0], top_k=5)


hit1 = hit3 = 0
answerable_tops: list[float] = []
for question, expected in ANSWERABLE:
    hits = top_hits(question)
    sops = [h.chunk.metadata["sop_id"] for h in hits]
    rank = sops.index(expected) + 1 if expected in sops else None
    hit1 += rank == 1
    hit3 += rank is not None and rank <= 3
    answerable_tops.append(hits[0].score)
    print(f"rank={rank}  top={hits[0].score:.3f}  {question}")

print(f"\nhit@1: {hit1}/{len(ANSWERABLE)}   hit@3: {hit3}/{len(ANSWERABLE)}")
print(f"lowest top score on answerable questions: {min(answerable_tops):.3f}")

print("\nUnanswerable questions:")
unanswerable_tops = []
for question in UNANSWERABLE:
    hits = top_hits(question)
    unanswerable_tops.append(hits[0].score)
    print(f"top={hits[0].score:.3f}  {question}")
print(f"highest top score on unanswerable questions: {max(unanswerable_tops):.3f}")