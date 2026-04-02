from mpmath import li

import os
import json
import asyncio
from pathlib import Path
from datasets import Dataset, load_from_disk
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
from loguru import logger
from lcr.formatter import DataFormatter

class QueryGenerator:
    """
    Class responsible for generating queries given a chunk and its context.
    """
    def __init__(self, ds_formatter: DataFormatter, llm_name: str, save_path: str, start_from_checkpoint: bool = False, save_jsonl: bool = True):
        self.ds_formatter = ds_formatter
        self.llm_name = llm_name
        self.save_path = Path(save_path)
        self.start_from_checkpoint = start_from_checkpoint
        self.save_jsonl = save_jsonl
        self._semaphore = asyncio.Semaphore(50)
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.queries = []

    def _get_prompt(self, chunk: str, context_chunks: str) -> str:
        # User should edit this prompt as needed
        return f"Write a query in Polish concerning the Chunk: {chunk}\n that requires the Context: {context_chunks}. Output the query without any additional text."

    async def _generate_query(self, chunk: str, context_chunks: str) -> str:
        prompt = self._get_prompt(chunk, context_chunks)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.llm_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.0,
                        extra_body={"reasoning": {"enabled": False}},
                    )
                choices = getattr(response, "choices", None)
                if not choices or not hasattr(choices[0], "message"):
                    raise ValueError("No choices/message in response")
                content = getattr(choices[0].message, "content", None)
                if not content or not content.strip():
                    raise ValueError("Empty content in response")
                return content.strip()
            except Exception as e:
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
                    wait_time = 2 ** (attempt - 1)
                    logger.warning(f"[Retry {attempt}/{max_retries}] Query generation failed due to retryable error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[Skip] Query generation failed for chunk after {attempt} attempts. Error: {e}")
                    break
        logger.error("[Failure] Marking query as <QUERY_GENERATION_FAILURE>.")
        return "<QUERY_GENERATION_FAILURE>"

    def _load_existing(self):
        """Load existing queries from HuggingFace dataset if checkpointing."""
        if self.save_path.exists():
            try:
                ds = load_from_disk(str(self.save_path))
                logger.info(f"Loaded {len(ds)} queries from checkpoint at {self.save_path}")
                return ds
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {self.save_path}: {e}")
        return []

    def _save(self):
        """Save queries to HuggingFace dataset and optionally JSONL."""
        if not self.queries:
            return
        # Save as HuggingFace dataset
        data = {
            "chunk_id": [q["chunk_id"] for q in self.queries],
            # "chunk": [q["chunk"] for q in self.queries],
            # "context_chunks": [q["context_chunks"] for q in self.queries],
            "query": [q["query"] for q in self.queries],
        }
        ds = Dataset.from_dict(data)
        ds.save_to_disk(str(self.save_path))

        if self.save_jsonl:
            jsonl_path = self.save_path.parent / "queries.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for q in self.queries:
                    json.dump(q, f, ensure_ascii=False)
                    f.write("\n")
        logger.info(f"Saved {len(self.queries)} queries to {self.save_path} (and JSONL: {self.save_jsonl})")

    async def generate_queries(self):
        # Load checkpoint if needed
        existing = []
        existing_ids = set()
        if self.start_from_checkpoint:
            existing = self._load_existing()
            existing_ids = existing['chunk_id'] # they are already unique
            self.queries = list(existing)
        else:
            self.queries = []

        # Each pair: (chunk, context_chunks)
        # Assign a chunk_id (could be index or hash)
        pairs_with_id = []
        for chunk_id, chunk, context_chunks in self.ds_formatter.get_chunks_with_context():
            if chunk_id not in existing_ids:
                pairs_with_id.append((chunk_id, chunk, context_chunks))

        total = len(pairs_with_id)
        logger.info(f"Generating queries for {total} chunks (skipping {len(existing_ids)} already processed)")

        for i, (chunk_id, chunk, context_chunks) in enumerate(atqdm(pairs_with_id, desc="Generating queries", unit="chunk")):
            query = await self._generate_query(chunk, context_chunks)
            result = {
                "chunk_id": chunk_id,
                # "chunk": chunk,
                # "context_chunks": context_chunks,
                "query": query,
            }
            self.queries.append(result)
            if (i + 1) % 10 == 0:
                self._save()
        # Final save
        self._save()
        logger.info("Done generating queries.")

    