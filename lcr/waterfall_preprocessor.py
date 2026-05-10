import asyncio
import json
from typing import Literal

from loguru import logger
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoTokenizer

from lcr.anthropic_preprocessor import AnthropicContextualPreprocessor
from lcr.formatter import DataFormatter


class WaterfallContextualPreprocessor(AnthropicContextualPreprocessor):
    """
    Waterfall variant of `AnthropicContextualPreprocessor`.

    Splits each document into overlapping token windows and contextualises every
    chunk against every window. The per-window contexts are aggregated either by
    appending or by a second LLM consolidation pass.
    """

    def __init__(
        self,
        data_formatter: DataFormatter,
        contextualisation_model: str,
        provider: str = "vllm",
        max_concurrent: int = 64,
        start_from_checkpoint: bool = False,
        save_dir: str = "temp_augmented_docs",
        window_tokens: int = 32_000,
        overlap_tokens: int = 8_000,
        aggregation: Literal["append", "consolidate"] = "append",
        tokenizer_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    ):
        super().__init__(
            data_formatter=data_formatter,
            contextualisation_model=contextualisation_model,
            provider=provider,
            max_concurrent=max_concurrent,
            start_from_checkpoint=start_from_checkpoint,
            save_dir=save_dir,
        )
        if overlap_tokens >= window_tokens:
            raise ValueError("overlap_tokens must be smaller than window_tokens")
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.aggregation = aggregation
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.failure_mode: Literal["all", "any"] = "all" # if "all", only return failure signal if all windows fail; if "any", return failure signal if any window fails

    def _get_contextualisation_prompt(self, document: str, chunk: str) -> str:
        return (
            "System: You are a precise context augmenter \n"
            "User:\n"
            "Goal: Give a context to situate this chunk in the context of the document for the purposes of improving search retrieval of the chunk\n"
            "Instructions: \n"
            "Please give a succinct context to situate this chunk within the "
            "overall document for the purposes of improving search retrieval of the chunk. "
            "Answer only with the context and nothing else."
            "Note: you are given a part of the document, not the whole document. Provide context based only on this part.\n"
            "Context: \n"
            f"<document_part>\n{document}\n</document_part>\n"
            f"Here is the chunk we want to situate within the whole document\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Instructions reminder:\n"
            "Please give a succinct context to situate this chunk within the "
            "overall document for the purposes of improving search retrieval of the chunk. "
            "Answer only with the context and nothing else."
            "Answer in the language of the document. (Document and the chunk are in the same language)"
        )

    def _split_into_windows(self, document: str) -> list[str]:
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        if len(tokens) <= self.window_tokens:
            return [document]
        step = self.window_tokens - self.overlap_tokens
        windows: list[str] = []
        for start in range(0, len(tokens), step):
            window_tokens = tokens[start : start + self.window_tokens]
            windows.append(self.tokenizer.decode(window_tokens, skip_special_tokens=True))
            if start + self.window_tokens >= len(tokens):
                break
        return windows

    def _consolidation_prompt(self, contexts: list[str], chunk: str) -> str:
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(contexts))
        return (
            "System: You are a precise context augmenter\n"
            "User:\n"
            "Goal: You are given several context snippets that were each written to situate the same chunk "
            "within different parts of a long document. Merge them into one succinct context that captures "
            "all distinct information, removes duplication, and is suitable for improving search retrieval.\n"
            "Instructions:\n"
            "Answer only with the merged context and nothing else.\n"
            f"<contexts>\n{numbered}\n</contexts>\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            "Answer in the language of the chunk."
        )

    async def _consolidate_contexts(self, contexts: list[str], chunk: str, cache_id: str) -> str:
        prompt = self._consolidation_prompt(contexts, chunk)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.contextualisation_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        extra_body={
                            "reasoning": {"effort": "low"},
                            "provider": {"order": ["deepinfra"], "allow_fallbacks": False},
                            "prompt_cache_key": cache_id,
                        },
                    )
                usage = getattr(response, "usage", None)
                if usage:
                    cost = getattr(usage, "cost", None)
                    self.total_cost += cost if cost else 0.0

                choices = getattr(response, "choices", None)
                if not choices or not hasattr(choices[0], "message"):
                    raise ValueError("No choices/message in consolidation response")
                content = getattr(choices[0].message, "content", None)
                if not content or not content.strip():
                    raise ValueError("Empty content in consolidation response")
                return content.strip() + "\n\n"
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** (attempt - 1)
                    logger.warning(
                        f"[Consolidate retry {attempt}/{max_retries}] cache_id={cache_id}: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"[Consolidate failure] cache_id={cache_id}: all {max_retries} attempts failed. Last error: {e}"
                    )
        return "<CONSOLIDATION_FAILURE>"

    async def _aggregate(self, contexts: list[str], chunk: str, cache_id: str) -> str:
        """By default: if all contexts are empty or failed, return a failure signal."""
        valid = [c for c in contexts if c and not c.startswith("<CONTEXTUALISATION_FAILURE")]
        if self.failure_mode == "any" and len(valid) < len(contexts):
            return "<CONTEXTUALISATION_FAILURE>"
        if self.failure_mode == "all" and len(valid) == 0:
            return "<CONTEXTUALISATION_FAILURE>" 
        if self.aggregation == "append":
            return "\n\n".join(c.strip() for c in valid) + "\n\n"
        return await self._consolidate_contexts(valid, chunk, cache_id=cache_id)

    async def _waterfall_chunk(
        self,
        windows: list[str],
        chunk: str,
        chunk_id: str,
        doc_id: str,
    ) -> tuple[str, str, str]:
        tasks = [
            self._contextualise_chunk(window, chunk, cache_id=f"{doc_id}_w{w_idx}")
            for w_idx, window in enumerate(windows)
        ]
        contexts = await asyncio.gather(*tasks)
        aggregated = await self._aggregate(contexts, chunk, cache_id=f"{doc_id}_consolidate")
        return chunk_id, chunk, aggregated

    async def augment_documents(self, col: str = "chunk") -> None:
        await self.double_check()

        existing_chunk_ids: set[str] = set()
        if self.start_from_checkpoint:
            existing_ds = self._load_existing()
            self.augmented_documents = existing_ds
            existing_chunk_ids = set(doc["chunk_id"] for doc in existing_ds)
            logger.info(
                f"Resuming augmentation. {len(existing_chunk_ids)} chunks already contextualised and will be skipped."
            )

        chunk_groups, chunks_ids = self.data_formatter.get_nested(col=col)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        per_chunk_tasks = []
        for i, chunk_group in enumerate(chunk_groups):
            merged_doc = "\n".join(chunk_group)
            windows = self._split_into_windows(merged_doc)
            doc_id = chunks_ids[i][0].split("_")[0] if chunks_ids[i] else f"doc{i}"
            logger.info(
                f"Document {doc_id}: {len(windows)} windows (window_tokens={self.window_tokens}, "
                f"overlap_tokens={self.overlap_tokens}), {len(chunk_group)} chunks"
            )
            for j, chunk in enumerate(chunk_group):
                chunk_id = chunks_ids[i][j]
                if chunk_id in existing_chunk_ids:
                    continue
                per_chunk_tasks.append(
                    self._waterfall_chunk(windows, chunk, chunk_id, doc_id)
                )

        total = len(per_chunk_tasks)
        print(f"Waterfall-contextualising {total} chunks across {len(chunk_groups)} documents...")

        completed = 0
        for coro in atqdm(
            asyncio.as_completed(per_chunk_tasks),
            total=total,
            desc="Waterfall contextualising chunks",
            unit="chunk",
        ):
            chunk_id, chunk, aggregated = await coro
            self.augmented_documents.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": aggregated + " " + chunk,
                }
            )
            completed += 1
            atqdm.write(f"Completed {completed}/{total} contextualisations")
            if completed % 10 == 0 or completed == total:
                logger.info(
                    f"Cost so far: ${self.total_cost:.4f} for {completed} contextualisations. Saving progress..."
                )
                self.save_augmented_documents(self.save_dir)

        logger.info(
            f"Done. {total} chunks contextualised. Total API cost: ${self.total_cost:.4f}"
        )
