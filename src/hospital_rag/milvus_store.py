from pymilvus import DataType, MilvusClient

from src.hospital_rag.interfaces import Chunk, SearchHit


class MilvusVectorStore:
    def __init__(
        self,
        dimension: int,
        collection_name: str = "hospital_chunks",
        uri: str = "http://localhost:19530",
    ) -> None:
        self._client = MilvusClient(uri=uri)
        self._collection = collection_name
        self._dimension = dimension
        if not self._client.has_collection(collection_name):
            self._create_collection()

    def _create_collection(self) -> None:
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dimension)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        rows = [
            {
                "id": c.id,
                "document_id": c.document_id,
                "text": c.text,
                "metadata": c.metadata,
                "vector": v,
            }
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, data=rows)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchHit]:
        results = self._client.search(
            collection_name=self._collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["document_id", "text", "metadata"],
        )
        hits: list[SearchHit] = []
        for hit in results[0]:
            entity = hit["entity"]
            chunk = Chunk(
                id=hit["id"],
                document_id=entity["document_id"],
                text=entity["text"],
                metadata=entity["metadata"],
            )
            hits.append(SearchHit(chunk=chunk, score=float(hit["distance"])))
        return hits

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            filter="document_id == {doc_id}",
            filter_params={"doc_id": document_id},
        )

    def drop(self) -> None:
        self._client.drop_collection(self._collection)