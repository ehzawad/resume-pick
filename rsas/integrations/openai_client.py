"""OpenAI client wrapper with retry logic and Response API support."""

import hashlib
import os
import json
from pathlib import Path
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..observability.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Wrapper for OpenAI API with retry logic and structured outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.1",
        reasoning_effort: str = "high",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Default model to use
            reasoning_effort: Reasoning effort level (low, medium, high) mapped to Responses API `reasoning.effort`
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.test_mode = bool(os.getenv("RSAS_TEST_MODE") or os.getenv("RSAS_MOCK_OPENAI"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key and not self.test_mode:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = None
        if not self.test_mode:
            # Initialize async client
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=max_retries,
            )

        logger.info("openai_client_initialized", model=model, reasoning_effort=reasoning_effort, test_mode=self.test_mode)

        # Lightweight local embedding cache
        self._embedding_cache_path = Path("data/cache/embeddings_cache.json")
        self._embedding_cache: dict[str, list[float]] = {}
        self._load_embedding_cache()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True,
    )
    async def create_response(
        self,
        input_text: Any,
        response_model: Type[T],
        model: str | None = None,
        reasoning_effort: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[T, dict[str, Any]]:
        """Create a response using OpenAI Response API with structured output.

        Args:
            input_text: Input prompt text
            response_model: Pydantic model for structured output
            model: Model to use (defaults to instance default)
            reasoning_effort: Reasoning effort (defaults to instance default)
            metadata: Additional metadata for the request
            max_output_tokens: Optional cap on generated tokens

        Returns:
            Tuple of (parsed response, metadata including tokens used)

        Raises:
            OpenAIError: If API call fails after retries
        """
        model = model or self.model
        reasoning_effort = reasoning_effort or self.reasoning_effort

        if self.test_mode:
            fabricated = self._fabricate_model(response_model)
            return fabricated, {"tokens_total": 0, "model": model, "mock": True}

        logger.info(
            "creating_response",
            model=model,
            reasoning_effort=reasoning_effort,
            input_length=len(str(input_text)),
        )

        try:
            # Build JSON schema for strict structured output (Responses API)
            schema = response_model.model_json_schema()
            self._enforce_no_extra_properties(schema)

            schema_hint = json.dumps(schema)
            text_format = {"type": "json_object"}

            # Normalize input into Responses API shape
            # Ensure input contains a JSON directive so the API honors json_object formatting
            if isinstance(input_text, str):
                input_payload = [
                    {
                        "role": "system",
                        "content": "Return a JSON object that follows the provided schema guidance.",
                    },
                    {"role": "user", "content": input_text},
                ]
            elif isinstance(input_text, list):
                input_payload = [
                    {
                        "role": "system",
                        "content": "Return a JSON object that follows the provided schema guidance.",
                    }
                ] + list(input_text)
            else:
                input_payload = input_text

            reasoning_payload = None  # Reasoning tokens can consume budget before output; disable by default

            completion = await self.client.responses.create(
                model=model,
                input=input_payload,
                instructions=(
                    "You are a structured extraction model. "
                    "Return ONLY a valid JSON object that conforms to the provided schema. "
                    "Do not add markdown, code fences, prose, or extra keys. "
                    "If a field is unknown, set it to null or an empty list/object. "
                    "Limit lists to the top 5 items and keep text concise. "
                    "If you cannot comply, return an empty JSON object. "
                    f"Schema (JSON): {schema_hint}"
                ),
                max_output_tokens=max_output_tokens or 4096,
                metadata=metadata or {},
                reasoning=reasoning_payload,
                text={"format": text_format},
            )

            output_text = self._extract_text_from_response(completion)
            if not output_text:
                raise ValueError("No text content returned from OpenAI response")

            parsed_output = self._parse_response_json(response_model, output_text)

            usage_metadata = {
                "tokens_total": getattr(completion.usage, "total_tokens", 0),
                "tokens_input": getattr(completion.usage, "input_tokens", 0)
                or getattr(completion.usage, "prompt_tokens", 0),
                "tokens_output": getattr(completion.usage, "output_tokens", 0)
                or getattr(completion.usage, "completion_tokens", 0),
                "response_id": getattr(completion, "id", None),
                "model": getattr(completion, "model", model),
            }

            logger.info(
                "response_created",
                response_id=getattr(completion, "id", None),
                tokens_total=usage_metadata["tokens_total"],
            )

            return parsed_output, usage_metadata

        except OpenAIError as e:
            logger.error(
                "openai_error",
                error=str(e),
                model=model,
                exc_info=True,
            )
            raise

    def _parse_response_json(self, response_model: Type[T], output_text: str) -> T:
        """Parse model output into the response model with basic JSON repair."""
        try:
            return response_model.model_validate_json(output_text)
        except Exception:
            repaired = self._repair_json(output_text)
            if repaired is not None:
                try:
                    return response_model.model_validate(repaired)
                except Exception:
                    pass
            # Re-raise original failure to surface parsing error
            raise

    def _repair_json(self, text: str) -> dict[str, Any] | None:
        """Best-effort JSON repair: trim to outermost braces and parse."""
        if "{" not in text or "}" not in text:
            return None
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True,
    )
    async def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        """Generate text embedding using OpenAI embeddings API.

        Args:
            text: Text to embed
            model: Embedding model to use (default: text-embedding-3-small)
            dimensions: Optional dimensions for embedding (for compatible models)

        Returns:
            Tuple of (embedding vector, metadata with usage info)

        Raises:
            OpenAIError: If API call fails after retries
        """
        logger.info("generating_embedding", model=model, text_length=len(text), test_mode=self.test_mode)

        if self.test_mode:
            dim = dimensions or 8
            embedding = [0.0] * dim
            return embedding, {"tokens_used": 0, "model": model, "dimensions": dim, "mock": True}

        try:
            # Call OpenAI embeddings API
            kwargs = {"model": model, "input": text}
            if dimensions:
                kwargs["dimensions"] = dimensions

            cache_key = self._cache_key(text, model, dimensions)
            if cache_key in self._embedding_cache:
                embedding = self._embedding_cache[cache_key]
                return embedding, {"tokens_used": 0, "model": model, "dimensions": len(embedding), "cached": True}

            response = await self.client.embeddings.create(**kwargs)

            # Extract embedding vector
            embedding = response.data[0].embedding
            # store in cache
            self._embedding_cache[cache_key] = embedding
            self._persist_embedding_cache()

            # Extract usage metadata
            usage_metadata = {
                "tokens_used": response.usage.total_tokens,
                "model": model,
                "dimensions": len(embedding),
            }

            logger.info(
                "embedding_generated",
                tokens=usage_metadata["tokens_used"],
                dimensions=usage_metadata["dimensions"],
            )

            return embedding, usage_metadata

        except OpenAIError as e:
            logger.error("openai_embedding_error", error=str(e), model=model, exc_info=True)
            raise

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 100,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            dimensions: Optional dimensions for embedding
            batch_size: Maximum texts per batch (OpenAI limit is 2048)

        Returns:
            Tuple of (list of embedding vectors, aggregated metadata)

        Raises:
            OpenAIError: If API call fails after retries
        """
        logger.info("generating_embeddings_batch", count=len(texts), batch_size=batch_size, test_mode=self.test_mode)

        if self.test_mode:
            dim = dimensions or 8
            embeddings = [[0.0] * dim for _ in texts]
            return embeddings, {
                "tokens_used": 0,
                "texts_count": len(texts),
                "model": model,
                "dimensions": dim,
                "mock": True,
            }

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                kwargs = {"model": model, "input": batch}
                if dimensions:
                    kwargs["dimensions"] = dimensions

                # Check cache and only call API for missing items
                batch_embeddings: list[list[float]] = []
                to_fetch_indices: list[int] = []
                to_fetch_texts: list[str] = []
                for idx, txt in enumerate(batch):
                    key = self._cache_key(txt, model, dimensions)
                    if key in self._embedding_cache:
                        batch_embeddings.append(self._embedding_cache[key])
                    else:
                        to_fetch_indices.append(idx)
                        to_fetch_texts.append(txt)
                        batch_embeddings.append([])  # placeholder

                if to_fetch_texts:
                    response = await self.client.embeddings.create(
                        model=model,
                        input=to_fetch_texts,
                        **({"dimensions": dimensions} if dimensions else {}),
                    )
                    total_tokens += response.usage.total_tokens
                    fetched = [item.embedding for item in response.data]
                    for offset, embed in zip(to_fetch_indices, fetched):
                        full_idx = offset
                        batch_embeddings[full_idx] = embed
                        key = self._cache_key(batch[full_idx], model, dimensions)
                        self._embedding_cache[key] = embed
                    self._persist_embedding_cache()

                all_embeddings.extend(batch_embeddings)

                logger.info(
                    "batch_embedded",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                    tokens=response.usage.total_tokens,
                )

            except OpenAIError as e:
                logger.error(
                    "batch_embedding_error",
                    batch_num=i // batch_size + 1,
                    error=str(e),
                    exc_info=True,
                )
                raise

        usage_metadata = {
            "tokens_used": total_tokens,
            "texts_count": len(texts),
            "model": model,
            "dimensions": len(all_embeddings[0]) if all_embeddings else 0,
        }

        logger.info(
            "embeddings_batch_complete",
            total_texts=len(texts),
            total_tokens=total_tokens,
        )

        return all_embeddings, usage_metadata

    @staticmethod
    def hash_input(input_data: str | dict[str, Any]) -> str:
        """Create SHA-256 hash of input for idempotency.

        Args:
            input_data: Input string or dictionary

        Returns:
            Hexadecimal hash string
        """
        if isinstance(input_data, dict):
            # Sort keys for consistent hashing
            import json

            input_str = json.dumps(input_data, sort_keys=True)
        else:
            input_str = input_data

        return hashlib.sha256(input_str.encode()).hexdigest()

    def _fabricate_model(self, model_cls: Type[T]) -> T:
        """Create a minimal instance of a Pydantic model for test mode."""
        from pydantic import BaseModel
        from enum import Enum

        def fabricate_value(annotation, field_info):
            if field_info.default is not None and field_info.default is not field_info.default_factory:
                return field_info.default
            if field_info.default_factory:
                try:
                    return field_info.default_factory()
                except Exception:
                    return None
            origin = getattr(annotation, "__origin__", None)
            args = getattr(annotation, "__args__", ())
            if origin in (list, list.__class__):
                return []
            if origin is list:
                return []
            if origin is dict:
                return {}
            if annotation is str:
                return ""
            if annotation is int:
                return 0
            if annotation is float:
                return 0.0
            if annotation is bool:
                return False
            if isinstance(annotation, type) and issubclass(annotation, Enum):
                return list(annotation)[0].value
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return self._fabricate_model(annotation)
            return None

        if issubclass(model_cls, BaseModel):
            values = {}
            for name, field in model_cls.model_fields.items():
                values[name] = fabricate_value(field.annotation, field)
            return model_cls.model_validate(values)
        # fallback for non-pydantic
        return model_cls()  # type: ignore

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        """Extract concatenated text payload from a Responses API result."""
        # Prefer helper if present
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text)

        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_val = getattr(content, "text", None)
                if text_val:
                    texts.append(str(text_val))
        if not texts:
            logger.error(
                "empty_response_output",
                status=getattr(response, "status", None),
                output_preview=str(getattr(response, "output", None))[:500],
                output_text=getattr(response, "output_text", None),
            )
        return "".join(texts).strip()

    def _enforce_no_extra_properties(self, schema: JsonSchemaValue) -> None:
        """Ensure JSON schema objects forbid additional properties (Responses API requirement)."""

        def recurse(node: Any) -> None:
            if isinstance(node, dict):
                if "$ref" in node:
                    # Responses API rejects sibling keywords alongside $ref
                    for key in list(node.keys()):
                        if key != "$ref":
                            node.pop(key, None)
                    return
                if node.get("type") == "object":
                    node.setdefault("additionalProperties", False)
                    if node.get("properties"):
                        node["required"] = sorted(node["properties"].keys())
                    for prop in node.get("properties", {}).values():
                        recurse(prop)
                    if "items" in node:
                        recurse(node["items"])
                    for key in ("allOf", "anyOf", "oneOf"):
                        if key in node and isinstance(node[key], list):
                            for sub in node[key]:
                                recurse(sub)
                elif "items" in node:
                    recurse(node["items"])
                for value in node.values():
                    recurse(value)
            elif isinstance(node, list):
                for item in node:
                    recurse(item)

        recurse(schema)

    @staticmethod
    def _supports_reasoning(model: str) -> bool:
        """Return True if the model supports the Responses API reasoning block."""
        normalized = model.lower()
        return normalized.startswith("gpt-5") or normalized.startswith("o")

    def _cache_key(self, text: str, model: str, dimensions: int | None) -> str:
        return self.hash_input({"text": text, "model": model, "dimensions": dimensions or "full"})

    def _load_embedding_cache(self) -> None:
        try:
            if self._embedding_cache_path.exists():
                self._embedding_cache = json.loads(self._embedding_cache_path.read_text())
        except Exception as exc:
            logger.error("embedding_cache_load_failed", error=str(exc), path=str(self._embedding_cache_path))

    def _persist_embedding_cache(self) -> None:
        try:
            self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._embedding_cache_path.write_text(json.dumps(self._embedding_cache))
        except Exception as exc:
            logger.error("embedding_cache_persist_failed", error=str(exc), path=str(self._embedding_cache_path))


# Global client instance
_openai_client: OpenAIClient | None = None


def get_openai_client(config: dict[str, Any] | None = None) -> OpenAIClient:
    """Get global OpenAI client instance.

    Args:
        config: Optional configuration dict

    Returns:
        OpenAIClient instance
    """
    global _openai_client

    if _openai_client is None:
        if config:
            openai_config = config.get("openai", {})
            _openai_client = OpenAIClient(
                model=openai_config.get("model", "gpt-5.1"),
                reasoning_effort=openai_config.get("reasoning_effort", "high"),
                timeout=openai_config.get("timeout", 60),
                max_retries=openai_config.get("max_retries", 3),
            )
        else:
            # Use defaults
            _openai_client = OpenAIClient()

    return _openai_client
