import asyncio
import json
import os
from pathlib import Path

from datasets import Dataset
import dotenv
from loguru import logger
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
            "Answer in the language of the document. (Document and the chunk are in the same language)"
            # I think it's fine - let's go.
        )

    async def _contextualise_chunk(self, document: str, chunk: str) -> str:
        """Call OpenRouter async, respecting the concurrency semaphore, with retries."""
        prompt = self._get_contextualisation_prompt(document, chunk)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.contextualisation_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400,
                        temperature=0.0,  # deterministic for retrieval tasks
                        extra_body={"reasoning": {"enabled": False}},
                    )
                # Defensive: check response structure and content
                choices = getattr(response, "choices", None)
                if not choices or not hasattr(choices[0], "message"):
                    raise ValueError("No choices/message in response")
                content = getattr(choices[0].message, "content", None)
                if not content or not content.strip():
                    raise ValueError("Empty content in response")
                context = content.strip()
                # logger.info(f"Contextualised successfully on attempt {attempt}.")
                # logger.debug(f"Chunk: {chunk}")
                # logger.debug(f"Contextualised chunk: {context}")
                return context
            except Exception as e:
                # Identify retryable errors (rate limit, network, etc.)
                err_str = str(e).lower()
                is_retryable = (
                    "rate limit" in err_str or
                    "429" in err_str or
                    "timeout" in err_str or
                    "temporarily unavailable" in err_str or
                    "connection" in err_str or
                    "network" in err_str or
                    "service unavailable" in err_str
                )
                if attempt < max_retries and is_retryable:
                # if attempt < max_retries:
                    wait_time = 2 ** (attempt - 1)
                    logger.warning(f"[Retry {attempt}/{max_retries}] Contextualisation failed due to retryable error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[Skip] Contextualisation failed for chunk after {attempt} attempts. Error: {e}")
                    break
        # If we reach here, all attempts failed
        logger.error("[Failure] Marking chunk as <CONTEXTUALISATION_FAILURE>.")
        return "<CONTEXTUALISATION_FAILURE>"

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

        self.augmented_documents = [{"chunk_id": doc_id, "chunk": ctx + " " + chunk} for (_, chunk, doc_id), ctx in zip(triples, contexts)]

    # save as a hugging face dataset, as it's more flexible for downstream use cases

    def save_augmented_documents(self, path: Path | str) -> None:
        """Save the augmented file as a Hugging Face dataset with 'chunk' and 'chunk_id' columns."""

        if self.augmented_documents is None:
            raise ValueError("No augmented documents to save. Run augment_documents first.")
        
        # sort by chunk_id for easier inspection (optional)
        self.augmented_documents.sort(key=lambda x: x["chunk_id"])

        data = {
            "chunk_id": [doc["chunk_id"] for doc in self.augmented_documents],
            "chunk": [doc["chunk"] for doc in self.augmented_documents],
        }
        ds = Dataset.from_dict(data)
        ds.save_to_disk(path)
        # also create a jsonl for easier inspection/debugging
        jsonl_path = Path(path).parent / "augmented_documents.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for doc in self.augmented_documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Augmented documents saved to {path} and {jsonl_path}")
