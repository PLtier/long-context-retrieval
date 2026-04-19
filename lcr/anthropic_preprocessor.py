import asyncio
import json
import os
from pathlib import Path

from datasets import Dataset
from datasets.load import load_dataset
import dotenv
from loguru import logger
from openai import AsyncOpenAI
from prompt_toolkit.layout import D
from tqdm.asyncio import tqdm as atqdm

from lcr.formatter import DataFormatter

# import tiktoken

dotenv.load_dotenv()  # Load environment variables from .env file



class AnthropicContextualPreprocessor:
    """
    Preprocessor that uses an LLM API (OpenRouter or Together) to contextualise chunks.
    """

    PROVIDER_BASE_URLS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "together": "https://api.together.xyz/v1",
    }

    PROVIDER_API_KEYS = {
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
    }

    def __init__(
        self,
        data_formatter: DataFormatter,
        contextualisation_model: str,
        provider: str = "openrouter",
        max_concurrent: int = 50, #TODO: REMOVE IN PRODUCTION
        start_from_checkpoint: bool = False,
        save_dir: str = "temp_augmented_docs",
    ):
        self.data_formatter = data_formatter
        self.contextualisation_model = contextualisation_model
        self.provider = provider
        self.augmented_documents = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.start_from_checkpoint = start_from_checkpoint
        self.save_dir = Path(save_dir)

        # self.tokenizer = tiktoken.get_encoding("o200k_harmony")
        self.total_cost = 0.0


        if provider not in self.PROVIDER_BASE_URLS:
            raise ValueError(f"Unsupported provider: {provider}")
        api_key_env = self.PROVIDER_API_KEYS[provider]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key for provider '{provider}' not found in environment variable '{api_key_env}'")
        base_url = self.PROVIDER_BASE_URLS[provider]
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
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

    async def _contextualise_chunk(self, document: str, chunk: str, cache_id: str ="") -> str:
        """Call OpenRouter async, respecting the concurrency semaphore, with retries."""
        prompt = self._get_contextualisation_prompt(document, chunk)
        # print(f"Prompt tokens: {len(self.tokenizer.encode(prompt))}")  # Debug token count of the prompt
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.contextualisation_model,
                        messages=[{"role": "user", "content": prompt}],
                        # max_completion_tokens=512,  # Adjust as needed
                        temperature=0.0,  # deterministic for retrieval tasks
                        extra_body={"reasoning": {"effort": "low"}, "provider": {"order": ["deepinfra"], "allow_fallbacks": False}, "prompt_cache_key": cache_id}
                    )
                # Defensive: check response structure and content

                usage = getattr(response, "usage", None)
                if usage:
                    cost = getattr(usage, "cost", None)
                    self.total_cost += cost if cost else 0.0
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    reasoning_tokens = None
                    ctd = getattr(usage, "completion_tokens_details", None)
                    if ctd:
                        reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
                else:
                    cost = completion_tokens = reasoning_tokens = None
                
                with open(self.save_dir / "debug_response.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "cost": cost,
                        "reasoning_tokens": reasoning_tokens,
                        "completion_tokens": completion_tokens,
                        "cache_id": cache_id,
                    }, f, ensure_ascii=False, indent=4)
                
                # ct  = getattr(response, "completion_tokens", 0)
                # est = getattr(response, "estimated_cost", None)
                # cached = 0
                # write_cache = 0
                # if ptd is not None:
                #     cached = getattr(ptd, "cached_tokens", 0) or (ptd.get("cached_tokens") if isinstance(ptd, dict) else 0)
                #     write_cache = getattr(ptd, "cache_write_tokens", 0) or (ptd.get("cache_write_tokens") if isinstance(ptd, dict) else 0)
                # logger.info(f"Contextualisation API call successful on attempt {attempt}. Tokens - Prompt: {pt} (Cached: {cached}, Cache Write: {write_cache}), Completion: {ct}, Estimated Cost: {est}")
                # logger.info(f"Contextualisation API call successful on attempt {attempt}")
                # logger.info(f"Usage details: {u}")
                # logger.info(f"Prompt tokens details: {ptd}")
                # logger.info(f"Cache key used: {cache_id}")
                # print hash of the document and chunk for debugging
                # logger.debug(f"Document hash: {hash(document)}, Chunk hash: {hash(chunk)}")


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
                return context + "\n\n"
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


    def _load_existing(self) -> list[dict[str, str]]:
        """Loads existing augmented documents from the save_dir"""
        jsonl_path = self.save_dir / "augmented_chunks.jsonl"
        if jsonl_path.exists():
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    ds = [json.loads(line) for line in f]
                logger.info(f"Loaded existing augmented documents from {jsonl_path}. Resuming augmentation.")
                return ds
            except Exception as e:
                logger.error(f"Failed to load existing augmented documents from {jsonl_path}. Starting fresh. Error: {e}")
        return []

    async def augment_documents(self, col: str = "chunk") -> None:
        """
        Fan out ALL chunk-contextualisation calls concurrently,
        then flatten into (text, id) pairs.
        """

        existing_chunk_ids = set()
        if self.start_from_checkpoint:
            existing_ds = self._load_existing()
            self.augmented_documents = existing_ds
            existing_chunk_ids = set(doc["chunk_id"] for doc in existing_ds)
            logger.info(f"Resuming augmentation. {len(existing_chunk_ids)} chunks already contextualised and will be skipped.")

        chunks: list[list[str]]
        chunks_ids: list[list[str]]
        chunks, chunks_ids = self.data_formatter.get_nested(col=col)

        # Build one merged doc string per document group
        merged_documents = ["\n".join(chunks) for chunks in chunks]


        # Create a flat list of (merged_doc, chunk, doc_id) triples
        triples = [
            (merged_documents[i], chunk, chunks_ids[i][j])
            for i, chunk_group in enumerate(chunks)
            for j, chunk in enumerate(chunk_group)
            if chunks_ids[i][j] not in existing_chunk_ids  # Skip already contextualised chunks
        ]

        total = len(triples)
        print(f"Contextualising {total} chunks across {len(chunks)} documents...")

        # Fire all requests concurrently (semaphore handles back-pressure)
        tasks = [self._contextualise_chunk(merged_doc, chunk, cache_id = chunk_id.split("_")[0]) for merged_doc, chunk, chunk_id in triples]
        completed = 0
        for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Contextualising chunks", unit="chunk"):
            contextualisation = await coro  # Await each completed task to catch exceptions if needed

            doc_id = triples[completed][2]  # Get corresponding doc_id for this completed task
            chunk = triples[completed][1]  # Get corresponding chunk for this completed task
            self.augmented_documents.append(
                {
                    "chunk_id": doc_id,
                    "chunk": contextualisation + " " + chunk,  # Prepend context to original chunk
                }
            ) 

            completed += 1
            atqdm.write(f"Completed {completed}/{total} contextualisations")
            if completed % 10 == 0 or completed == total:
                logger.info(f"Cost so far: ${self.total_cost:.4f} for {completed} contextualisations. Saving progress...")
                self.save_augmented_documents(self.save_dir)  # Save progress every 10 chunks and at the end
        

        logger.info(f"Done. {total} chunks contextualised. Total API cost: ${self.total_cost:.4f}")


    # save as a hugging face dataset, as it's more flexible for downstream use cases

    def save_augmented_documents(self, path: Path | str) -> None:
        """Save the augmented file as a Hugging Face dataset with 'chunk' and 'chunk_id' columns."""

        if self.augmented_documents is None:
            raise ValueError("No augmented documents to save. Run augment_documents first.")
        
        # sort by chunk_id for easier inspection (optional)
        self.augmented_documents.sort(key=lambda x: x["chunk_id"])

        # data = {
        #     "chunk_id": [doc["chunk_id"] for doc in self.augmented_documents],
        #     "chunk": [doc["chunk"] for doc in self.augmented_documents],
        # }
        # ds = Dataset.from_dict(data)
        # ds.save_to_disk(path)
        # also create a jsonl for easier inspection/debugging
        Path(path).mkdir(parents=True, exist_ok=True)
        jsonl_path = Path(path) / "augmented_chunks.jsonl"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for doc in self.augmented_documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Augmented documents saved to {jsonl_path}")
