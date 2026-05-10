import asyncio
import json
import os
from pathlib import Path

from datasets import Dataset
from datasets.load import load_dataset
import dotenv
from loguru import logger
from openai import AsyncOpenAI
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
        "vllm": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    }

    PROVIDER_API_KEYS = {
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
        "vllm": None,
    }

    def __init__(
        self,
        data_formatter: DataFormatter,
        contextualisation_model: str,
        provider: str = "openrouter",
        max_concurrent: int = 64, #TODO: REMOVE IN PRODUCTION
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
        if api_key_env is None:
            api_key = "EMPTY"
        else:
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

    async def double_check(self) -> None:
        """Smoke-test the API: verifies auth, model availability, and prints context window."""
        logger.info(f"[SmokeTest] Checking model '{self.contextualisation_model}' on {self.provider}...")
        try:
            model_info = await self._client.models.retrieve(self.contextualisation_model)
            ctx = getattr(model_info, "context_length", None) or getattr(model_info, "context_window", None)
            logger.info(f"[SmokeTest] Context window: {f'{ctx:,} tokens' if ctx else 'not reported by API'}")
        except Exception:
            logger.info("[SmokeTest] Could not retrieve model info — skipping context window check")
        try:
            await self._client.chat.completions.create(
                model=self.contextualisation_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            logger.info("[SmokeTest] Auth and model OK.")
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                raise RuntimeError(f"[SmokeTest] Authentication failed for {self.provider}: {e}") from e
            elif "404" in err_str or "not found" in err_str:
                raise RuntimeError(f"[SmokeTest] Model '{self.contextualisation_model}' not found on {self.provider}: {e}") from e
            else:
                raise RuntimeError(f"[SmokeTest] Unexpected failure: {e}") from e

    async def _contextualise_chunk(self, document: str, chunk: str, cache_id: str = "") -> str:
        """Call OpenRouter async, respecting the concurrency semaphore, with retries."""
        prompt = self._get_contextualisation_prompt(document, chunk)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.contextualisation_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            extra_body={"reasoning": {"effort": "low"}, "provider": {"order": ["deepinfra"], "allow_fallbacks": False}, "prompt_cache_key": cache_id}
                        ),
                        timeout=600.0*3, # 30 minutes per request, it is likely a bug
                    )

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

                choices = getattr(response, "choices", None)
                if not choices or not hasattr(choices[0], "message"):
                    raise ValueError("No choices/message in response")
                content = getattr(choices[0].message, "content", None)
                if not content or not content.strip():
                    raise ValueError("Empty content in response")
                return content.strip() + "\n\n"

            except asyncio.TimeoutError:
                logger.error(f"[Timeout] cache_id={cache_id}: timed out after 10 minutes.")
                return "<CONTEXTUALISATION_FAILURE>"

            except Exception as e:
                err_str = str(e).lower()

                is_context_length = (
                    "context_length_exceeded" in err_str or
                    "maximum context length" in err_str or
                    "context window" in err_str or
                    "too long" in err_str
                )
                if is_context_length:
                    logger.critical(
                        f"\n{'='*60}\n"
                        f"CONTEXT LENGTH EXCEEDED — cache_id={cache_id}\n"
                        f"Document is too long for '{self.contextualisation_model}'.\n"
                        f"Skipping this chunk. Reduce document size or switch to a larger model.\n"
                        f"Error: {e}\n"
                        f"{'='*60}"
                    )
                    return "<CONTEXTUALISATION_FAILURE_CONTEXT_LENGTH>"

                is_fatal = any(code in err_str for code in ("401", "403", "404", "unauthorized", "forbidden", "not found"))
                if is_fatal:
                    logger.critical(
                        f"\n{'='*60}\n"
                        f"FATAL API ERROR — cache_id={cache_id}, attempt {attempt}\n"
                        f"Error: {e}\n"
                        f"{'='*60}"
                    )
                    return "<CONTEXTUALISATION_FAILURE>"

                if attempt < max_retries:
                    wait_time = 2 ** (attempt - 1)
                    logger.warning(f"[Retry {attempt}/{max_retries}] cache_id={cache_id}: {e}. {err_str} Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[Failure] cache_id={cache_id}: all {max_retries} attempts failed. Last error: {e}")

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
    async def _contextualise_with_id(self, merged_doc: str, chunk: str, chunk_id: str, doc_id: str) -> tuple[str, str, str]:
        ctx = await self._contextualise_chunk(merged_doc, chunk, cache_id=doc_id)
        return chunk_id, chunk, ctx
        

    async def augment_documents(self, col: str = "chunk") -> None:
        """
        Fan out ALL chunk-contextualisation calls concurrently,
        then flatten into (text, id) pairs.
        """
        await self.double_check()

        existing_chunk_ids = set()
        if self.start_from_checkpoint:
            existing_ds = self._load_existing()
            self.augmented_documents = existing_ds
            existing_chunk_ids = set(doc["chunk_id"] for doc in existing_ds)
            logger.info(f"Resuming augmentation. {len(existing_chunk_ids)} chunks already contextualised and will be skipped.")

        chunk_groups: list[list[str]]
        chunks_ids: list[list[str]]
        chunk_groups, chunks_ids = self.data_formatter.get_nested(col=col)

        # Build one merged doc string per document group
        merged_documents = ["\n".join(chunk_group) for chunk_group in chunk_groups]


        # Create a flat list of (merged_doc, chunk, chunk_id) triples
        triples = [
            (merged_documents[i], chunk, chunks_ids[i][j])
            for i, chunk_group in enumerate(chunk_groups)
            for j, chunk in enumerate(chunk_group)
            if chunks_ids[i][j] not in existing_chunk_ids  # Skip already contextualised chunks
        ]

        self.save_dir.mkdir(parents=True, exist_ok=True)

        total = len(triples)
        print(f"Contextualising {total} chunks across {len(chunk_groups)} documents...")
    

        # Fire all requests concurrently (semaphore handles back-pressure)
        tasks = [self._contextualise_with_id(merged_doc, chunk, chunk_id, chunk_id.split("_")[0]) for merged_doc, chunk, chunk_id in triples]
        completed = 0
        for coro in atqdm(asyncio.as_completed(tasks), total=total, desc="Contextualising chunks", unit="chunk"):
            chunk_id, chunk, contextualisation = await coro  # Await each completed task to catch exceptions if needed

            self.augmented_documents.append(
                {
                    "chunk_id": chunk_id,
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
