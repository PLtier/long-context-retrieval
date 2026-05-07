import asyncio
import json
import os
from pathlib import Path
import time

from datasets import Dataset, load_from_disk
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger
from openai import AsyncOpenAI
import pandas
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from lcr.config import PROMPTS_DIR
from lcr.formatter import DataFormatter


class QueryMapper:
    """
    Class responsible for generating queries given a chunk and its context.
    """
    def __init__(self, ds_formatter: DataFormatter, llm_name: str, provider: str, save_path: str, start_from_checkpoint: bool = False, save_jsonl: bool = True, context_col: str = "context_chunks_ids", impl_context_col: str = ""):
        self.ds_formatter: DataFormatter = ds_formatter
        
        self.llm_name: str = llm_name
        self.save_path: Path = Path(save_path)
        self.start_from_checkpoint: bool = start_from_checkpoint
        self.save_jsonl: bool = save_jsonl
        self.context_col: str = context_col
        self.impl_context_col: str = impl_context_col
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(64)
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "generic_response_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The generated result"
                        }
                    },
                    "required": ["result"],
                    "additionalProperties": False
                }
            }
        }

        if provider == "openrouter":
            self._client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        elif provider == "together":
            self._client = AsyncOpenAI(
                api_key=os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        elif provider == "vllm":
            self._client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.queries = []
    max_tokens = 200
    jsonl_filename = ""

    def _get_prompt(self, fields: dict[str, str]) -> str:
        raise NotImplementedError("Not implemented")

    async def _generate(self, fields: dict[str, str]) -> str:
        prompt = self._get_prompt(fields)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.llm_name,
                        # model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                        messages=[{"role": "user", "content": prompt}],
                        # max_tokens=self.max_tokens,
                        # temperature=0.0,
                        # extra_body={"reasoning": {"enabled": False}},
                        reasoning_effort=self.reasoning_effort,
                        # response_format={"type": "json_object"},
                        response_format=self.response_format,
                        extra_body={
                            "plugins": [{"id": "response-healing"}],
                            "reasoning": {"effort": self.reasoning_effort, "exclude": True},
                            "provider": {"require_parameters": True},
                        },
                        # reasoning = {"effort": "high"}
                    )
                choices = getattr(response, "choices", None)
                if not choices or not hasattr(choices[0], "message"):
                    raise ValueError("No choices/message in response")
                content = getattr(choices[0].message, "content", None)
                if not content or not content.strip():
                    logger.info(response)
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

    def _load_existing(self, jsonl_path: Path) -> list[dict]:
        """Load existing queries from jsonl dataset if checkpointing."""
        if jsonl_path.exists():
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    ds = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded existing augmented documents from {jsonl_path}. Resuming augmentation.")
                return ds
            except Exception as e:
                logger.error(f"Failed to load existing augmented documents from {jsonl_path}. Starting fresh. Error: {e}")
        return []

    def _save(self):
        """Save queries to HuggingFace dataset and optionally JSONL."""
        # i don't think we need to use it all all.
        # if not self.queries:
        #     return
        # # Save as HuggingFace dataset
        # keys = self.queries[0].keys()

        # data = {key: [q[key] for q in self.queries] for key in keys} # transform for HF format.
        # ds = Dataset.from_dict(data)
        # ds.save_to_disk(str(self.save_path))

        if self.save_jsonl:
            self.save_path.mkdir(parents=True, exist_ok=True)
            jsonl_path = self.save_path / f"{self.jsonl_filename}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for q in self.queries:
                    json.dump(q, f, ensure_ascii=False)
                    f.write("\n")
        logger.info(f"Saved {len(self.queries)} queries to {self.save_path} (and JSONL: {self.save_jsonl})")
    
    async def get_result(self, fields: dict[str,str]) -> dict[str, str]:
        raise NotImplementedError("Not implemented")

    


QUERY_SCHEMA_R8 = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_query_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "target_context_chunk_id": {
                    "type": "string",
                    "description": "The exact chunk_id of the targeted context chunk"
                },
                "query": {
                    "type": "string",
                    "description": "The generated retrieval query"
                }
            },
            "required": ["target_context_chunk_id", "query"],
            "additionalProperties": False
        }
    }
}

# Schema for query_prompt_r4.j2 (only 'query' key)
QUERY_SCHEMA_R4 = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_query_schema_r4",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The generated retrieval query"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}


QUERY_SCHEMA_R9 = {
    "type": "json_schema",
    "json_schema": {
        "name": "retrieval_query_schema_r9",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "utilized_context_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk_ids of the context chunks actually used to generate the query."
                },
                "query": {
                    "type": "string",
                    "description": "The generated retrieval query."
                }
            },
            "required": ["utilized_context_chunk_ids", "query"],
            "additionalProperties": False
        }
    }
}



class QueryGenerator(QueryMapper):
    """Generates queries given a chunk and its context."""
    def __init__(self, doc_formatter: DataFormatter, llm_name: str, provider: str, save_path: str, start_from_checkpoint: bool = False, save_jsonl: bool = True, context_col: str = "context_chunks_ids", impl_context_col: str = "", update_queries: bool = False, prompt_template: str = "query_prompt_r4.j2", input_queries_dir: str | None = None):
        super().__init__(doc_formatter, llm_name, provider, save_path, start_from_checkpoint, save_jsonl)
        self.max_tokens = 200
        self.jsonl_filename = "queries"
        self.context_col = context_col
        self.impl_context_col = impl_context_col
        self.update_queries = update_queries
        self.input_queries_dir = Path(input_queries_dir) if input_queries_dir else None
        self.prompt_template = prompt_template
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=select_autoescape()
        )
        self.template = self.jinja_env.get_template(self.prompt_template)
        # Select schema based on prompt template
        if "r4" in self.prompt_template:
            self.response_format = QUERY_SCHEMA_R4
        elif "r8" in self.prompt_template:
            self.response_format = QUERY_SCHEMA_R8
        elif "r9" in self.prompt_template:
            self.response_format = QUERY_SCHEMA_R9
        else:
            logger.error(f"No corresponding QUERY_SCHEMA for prompt template: {self.prompt_template}")
            raise ValueError(f"No corresponding QUERY_SCHEMA for prompt template: {self.prompt_template}")

    def _get_prompt(self, fields: dict[str, str]) -> str:
        # Render the prompt from the Jinja2 template
        return self.template.render(**fields)

    async def get_result(self, fields: dict[str, str]) -> dict[str, str]:
        raw = await self._generate(fields)
        try:
            parsed = json.loads(raw)  # guaranteed valid if json_schema worked
            obj = {
            "chunk_id": fields["chunk_id"],
            "query": parsed["query"],
            **fields,
            }
            if "target_context_chunk_id" in parsed:
                obj["target_context_chunk_id"] = parsed["target_context_chunk_id"]
            if "utilized_context_chunk_ids" in parsed:
                obj["utilized_context_chunk_ids"] = parsed["utilized_context_chunk_ids"]
            return obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {raw}. Error: {e}")
            obj ={
                "chunk_id": fields["chunk_id"],
                "query": raw,
                **fields,
            }
            if "target_context_chunk_id" in self.response_format["json_schema"]["schema"]["properties"]:
                obj["target_context_chunk_id"] = "<PARSING_FAILURE>"
            if "utilized_context_chunk_ids" in self.response_format["json_schema"]["schema"]["properties"]:
                obj["utilized_context_chunk_ids"] = []  # keep list type; string would break pyarrow schema inference
            return obj

    async def generate(self, chain_context: bool = True):
        # Load checkpoint if needed
        existing = []
        existing_ids = set()
        if self.start_from_checkpoint:
            load_path = self.input_queries_dir if self.update_queries and self.input_queries_dir else self.save_path
            existing = self._load_existing(load_path / f"{self.jsonl_filename}.jsonl")
            # print(existing[:2])
            existing_ids = set(chunk["chunk_id"] for chunk in existing) # they are already unique
            # print(existing_ids)
            # print(f"\n self.update_queries: {self.update_queries}")
            if not self.update_queries:
                self.queries = list(existing)
        else:
            self.queries = []

        # Each pair: (chunk, context_chunks)
        # Assign a chunk_id (could be index or hash)
        pairs_with_id = []
        print(existing_ids)
        print(f"\n self.update_queries: {self.update_queries}")
        print(self.ds_formatter.doc_dataset[:2])
        for chunk_id, chunk, context_chunks, impl_context_chunks in self.ds_formatter.get_chunks_with_context(chain_context=chain_context, context_col=self.context_col, impl_context_col=self.impl_context_col):
            # print(chunk_id)
            if chunk_id in existing_ids and self.update_queries:
                pairs_with_id.append((chunk_id, chunk, context_chunks, impl_context_chunks))
            elif chunk_id not in existing_ids and not self.update_queries:
                pairs_with_id.append((chunk_id, chunk, context_chunks, impl_context_chunks))
                


        total = len(pairs_with_id)
        logger.info(f"Generating queries for {total} chunks (skipping {len(existing_ids)} already processed)")

        start_time = time.time()
        tasks = []
        # i = 0
        for chunk_id, chunk, context_chunks, impl_context_chunks in tqdm(pairs_with_id, desc="Generating queries", unit="chunk"):
            fields = {
                "chunk_id": chunk_id,
                "chunk": chunk,
                "context_chunks": context_chunks,
                "impl_context_chunks": impl_context_chunks
            }
            # self.queries.append(fields) # TEST
            # TMP - limit
            # if i >= 30:
            #     break
            # else:
            #     i+=1
            tasks.append(self.get_result(fields))

        # # results = await asyncio.gather(*tasks)
        completed = 0

        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing generated queries", unit="query"):
            result = await coro
            self.queries.append(result)
            completed += 1
            if completed % 10 == 0:
                self._save()
        # # Final save
        self._save()
        end_time = time.time()
        logger.info(f"Done generating queries in {end_time - start_time:.2f} seconds.")

ASSURANCE_SCHEMA_R8 = {
    "type": "json_schema",
    "json_schema": {
        "name": "assurance_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "criterion_1": {"type": "string"},
                "criterion_2": {"type": "string"},
                "criterion_3": {"type": "string"},
                "criterion_4": {"type": "string"},
                "criterion_5": {"type": "string"},
                "criterion_6": {"type": "string"},
                "answer_to_query": {"type": "string"},
                "verdict": {"type": "string", "enum": ["Yes", "No"]}
            },
            "required": ["criterion_1","criterion_2","criterion_3","criterion_4","criterion_5","criterion_6","answer_to_query","verdict"],
            "additionalProperties": False
        }
    }
}

# Schema for assurance_prompt_r4.j2 (4 criteria)
ASSURANCE_SCHEMA_R4 = {
    "type": "json_schema",
    "json_schema": {
        "name": "assurance_result_r4",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "criterion_1": {"type": "string"},
                "criterion_2": {"type": "string"},
                "criterion_3": {"type": "string"},
                "criterion_4": {"type": "string"},
                "answer_to_query": {"type": "string"},
                "verdict": {"type": "string", "enum": ["Yes", "No"]}
            },
            "required": ["criterion_1","criterion_2","criterion_3","criterion_4","answer_to_query","verdict"],
            "additionalProperties": False
        }
    }
}

ASSURANCE_SCHEMA_R9 = {
    "type": "json_schema",
    "json_schema": {
        "name": "assurance_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "criterion_1": {"type": "string"},
                "criterion_2": {"type": "string"},
                "criterion_3": {"type": "string"},
                "criterion_4": {"type": "string"},
                "criterion_5": {"type": "string"},
                "answer_to_query": {"type": "string"},
                "verdict": {"type": "string", "enum": ["Yes", "No"]}
            },
            "required": ["criterion_1","criterion_2","criterion_3","criterion_4","criterion_5","answer_to_query","verdict"],
            "additionalProperties": False
        }
    }
}

ASSURANCE_SCHEMA_R10 = { # same as r4
    "type": "json_schema",
    "json_schema": {
        "name": "assurance_result_r10",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "criterion_1": {"type": "string"},
                "criterion_2": {"type": "string"},
                "criterion_3": {"type": "string"},
                "criterion_4": {"type": "string"},
                "answer_to_query": {"type": "string"},
                "verdict": {"type": "string", "enum": ["Yes", "No"]}
            },
            "required": ["criterion_1","criterion_2","criterion_3","criterion_4","answer_to_query","verdict"],
            "additionalProperties": False
        }
    }
}


class QueryAssurance(QueryMapper):
    """Goes through the generated queries and performs a check on them. Requires DataFormatter to have loaded queries."""
    def __init__(self, query_formatter: DataFormatter, llm_name: str, provider: str, save_path: str, start_from_checkpoint: bool = False, save_jsonl: bool = True, prompt_template: str = "assurance_prompt_r4.j2"):
        super().__init__(query_formatter, llm_name, provider, save_path, start_from_checkpoint, save_jsonl)
        self.max_tokens = 200
        self.jsonl_filename = "assurance_results"
        self.reasoning_effort = "high"
        self.prompt_template = prompt_template
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=select_autoescape()
        )
        self.template = self.jinja_env.get_template(self.prompt_template)
        # Select schema based on prompt template
        if "r4" in self.prompt_template:
            self.response_format = ASSURANCE_SCHEMA_R4
        elif "r8" in self.prompt_template:
            self.response_format = ASSURANCE_SCHEMA_R9
        elif "r9" in self.prompt_template:
            # For R9, we can reuse the same schema as R8 since the criteria are the same, just with an additional field for utilized context chunk ids. The prompt should be designed to include that in the response.
            self.response_format = ASSURANCE_SCHEMA_R9
        elif "r10" in self.prompt_template:
            # For R10, we can also reuse the same schema as R8, but we should ensure the prompt includes the context chunk ids in the response for better interpretability.
            self.response_format = ASSURANCE_SCHEMA_R10
        else:
            logger.error(f"No corresponding ASSURANCE_SCHEMA for prompt template: {self.prompt_template}")
            raise ValueError(f"No corresponding ASSURANCE_SCHEMA for prompt template: {self.prompt_template}")

    def _get_prompt(self, fields: dict[str, str]) -> str:
        # Render the prompt from the Jinja2 template
        return self.template.render(**fields)

    async def get_result(self, fields: dict[str, str]) -> dict[str, str]:
        raw = await self._generate(fields)
        try: 
            parsed = json.loads(raw)
            return {
                "chunk_id": fields["chunk_id"],
                "passes_assurance": parsed["verdict"],
                **{k: v for k, v in parsed.items() if k != "verdict"},
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {raw}. Error: {e}")
            return {
                "chunk_id": fields["chunk_id"],
                "passes_assurance": "<PARSING_FAILURE>" + raw,
            }

    async def generate(self):
        # Load checkpoint if needed
        existing_ids = set()
        if self.start_from_checkpoint:
            existing = self._load_existing(self.save_path / f"{self.jsonl_filename}.jsonl")
            # print(existing[:2])
            existing_ids = set(query['chunk_id'] for query in existing)  # they are already unique
            self.queries = list(existing)
        else:
            self.queries = []

        # Each pair: (chunk, context_chunks)
        # Assign a chunk_id (could be index or hash)
        pairs_with_id = []
        for row in self.ds_formatter.queries_dataset:
            chunk_id = row['chunk_id']
            chunk = row['chunk']
            context_chunks = row['context_chunks']
            impl_context_chunks = row.get('impl_context_chunks', "")
            utilized_context_chunk_ids = row.get('utilized_context_chunk_ids', "")
            query = row['query']
            if chunk_id not in existing_ids:
                pairs_with_id.append((chunk_id, query, chunk, context_chunks, impl_context_chunks, utilized_context_chunk_ids))

        total = len(pairs_with_id)
        logger.info(f"Generating queries for {total} chunks (skipping {len(existing_ids)} already processed)")

        tasks = []
        for chunk_id, query, chunk, context_chunks, impl_context_chunks, utilized_context_chunk_ids in tqdm(pairs_with_id, desc="Generating queries", unit="chunk"):
            fields = {
                "chunk_id": chunk_id,
                "query": query,
                "chunk": chunk,
                "context_chunks": context_chunks,
                "impl_context_chunks": impl_context_chunks,
                "context_chunk_ids": utilized_context_chunk_ids,
            }
            tasks.append(self.get_result(fields))

        completed = 0
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing generated queries", unit="query"):
            result = await coro
            self.queries.append(result)
            completed += 1
            if completed % 10 == 0:
                self._save()
        # Final save
        self._save()
        logger.info("Done generating queries.")

