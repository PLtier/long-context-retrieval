import asyncio
import os
from pathlib import Path

from datasets import Dataset
import dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from lcr.formatter import DataFormatter

dotenv.load_dotenv()  # Load environment variables from .env file


class AnthropicContextualPreprocessor:
    """
    Preprocessor that uses an OpenRouter model to contextualise chunks.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        data_formatter: DataFormatter,
        contextualisation_model: str,
        max_concurrent: int = 20,  # tune to your OpenRouter rate limit tier
    ):
        self.data_formatter = data_formatter
        self.contextualisation_model = contextualisation_model
        self.augmented_documents = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=self.OPENROUTER_BASE_URL,
        )

    def _get_contextualisation_prompt(self, document: str, chunk: str) -> str:
        return (
            "System: You are a precise context augmenter \n"
            "User:\n"
            "Goal: Give a context to situate this chunk in the context of the document for the purposes of improving search retrieval of the chunk\n"
            "Instructions: \n"
            "Please give a succinct context to situate this chunk within the "
            "overall document for the purposes of improving search retrieval of the chunk. "
            "Answer only with the context and nothing else."
            "Context: \n"
            f"<document>\n{document}\n</document>\n"
            f"Here is the chunk we want to situate within the whole document\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Instructions reminder:\n"
            "Please give a succinct context to situate this chunk within the "
            "overall document for the purposes of improving search retrieval of the chunk. "
            "Answer only with the context and nothing else."
            "Answer in Polish"
            # I think it's fine - let's go.
        )

    async def _contextualise_chunk(self, document: str, chunk: str) -> str:
        """Call OpenRouter async, respecting the concurrency semaphore."""
        prompt = self._get_contextualisation_prompt(document, chunk)
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.contextualisation_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,  # deterministic for retrieval tasks
            )
        context: str = response.choices[0].message.content.strip()  # ty:ignore[unresolved-attribute]
        print("====")
        # print(f"Chunk: {chunk}")
        # print(f"Contextualised chunk: {context}")  # Log the first 50 chars of the context
        return context

    async def augment_documents(self, col: str = "chunk") -> None:
        """
        Fan out ALL chunk-contextualisation calls concurrently,
        then flatten into (text, id) pairs.
        """
        documents, document_ids = self.data_formatter.get_nested(col=col)

        # Build one merged doc string per document group
        merged_documents = ["\n".join(chunks) for chunks in documents]


        # Create a flat list of (merged_doc, chunk, doc_id) triples
        triples = [
            (merged_documents[i], chunk, document_ids[i][j])
            for i, chunk_group in enumerate(documents)
            for j, chunk in enumerate(chunk_group)
        ]

        total = len(triples)
        print(f"Contextualising {total} chunks across {len(documents)} documents...")

        # Fire all requests concurrently (semaphore handles back-pressure)
        contexts = await atqdm.gather(
        *[self._contextualise_chunk(merged_doc, chunk) for merged_doc, chunk, _ in triples],
        desc="Contextualising chunks",
        unit="chunk",
        )

        print(f"Done. {total} chunks contextualised.")

        self.augmented_documents = [{"chunk": ctx + " " + chunk, "chunk_id": doc_id} for (_, chunk, doc_id), ctx in zip(triples, contexts)]

    # save as a hugging face dataset, as it's more flexible for downstream use cases

    def save_augmented_documents(self, path: Path | str) -> None:
        """Save the augmented file as a Hugging Face dataset with 'chunk' and 'chunk_id' columns."""

        if self.augmented_documents is None:
            raise ValueError("No augmented documents to save. Run augment_documents first.")
        data = {
            "chunk": [doc["chunk"] for doc in self.augmented_documents],
            "chunk_id": [doc["chunk_id"] for doc in self.augmented_documents],
        }
        ds = Dataset.from_dict(data)
        ds.save_to_disk(path)
