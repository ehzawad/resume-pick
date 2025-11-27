"""OpenAI client wrapper with retry logic and Response API support."""

import hashlib
import os
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
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
            reasoning_effort: Reasoning effort level (low, medium, high)
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True,
    )
    async def create_response(
        self,
        input_text: str,
        response_model: Type[T],
        model: str | None = None,
        reasoning_effort: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[T, dict[str, Any]]:
        """Create a response using OpenAI Response API with structured output.

        Args:
            input_text: Input prompt text
            response_model: Pydantic model for structured output
            model: Model to use (defaults to instance default)
            reasoning_effort: Reasoning effort (defaults to instance default)
            metadata: Additional metadata for the request

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
            input_length=len(input_text),
        )

        try:
            # Get JSON schema from Pydantic model
            schema = response_model.model_json_schema()

            # Enhance prompt to request JSON output
            enhanced_input = f"""{input_text}

IMPORTANT: Respond with valid JSON that matches this exact schema:
{schema}

Respond ONLY with the JSON object, no additional text."""

            # Call OpenAI Response API
            # Note: Response API does not support reasoning_effort parameter
            response = await self.client.responses.create(
                model=model,
                input=enhanced_input,
                metadata=metadata or {},
            )

            # Extract text output
            output_text = response.output[0].content[0].text if response.output else ""

            # Parse JSON from output
            import json
            try:
                # Find JSON in output (handle potential markdown code blocks)
                json_str = output_text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif json_str.startswith("```"):
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                # Parse to Pydantic model
                parsed_output = response_model.model_validate_json(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    "json_parse_error",
                    error=str(e),
                    output_text=output_text[:500],  # Log first 500 chars
                )
                raise ValueError(f"Failed to parse JSON from GPT-5.1 response: {e}")

            # Extract usage stats
            usage_metadata = {
                "tokens_total": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "tokens_input": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                "tokens_output": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                "response_id": response.id,
                "model": model,
            }

            logger.info(
                "response_created",
                response_id=response.id,
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
        reraise=True,
    )
    async def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: Type[T] | None = None,
    ) -> tuple[str | T, dict[str, Any]]:
        """Create a chat completion (fallback for non-Response API scenarios).

        Args:
            messages: List of message dicts with role and content
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional Pydantic model for structured output

        Returns:
            Tuple of (response content or parsed model, metadata)
        """
        model = model or self.model

        if self.test_mode:
            if response_format:
                return self._fabricate_model(response_format), {"tokens_total": 0, "model": model, "mock": True}
            return "mock-response", {"tokens_total": 0, "model": model, "mock": True}

        logger.info("creating_chat_completion", model=model, messages_count=len(messages))

        try:
            if response_format:
                # Use beta parse method for structured outputs
                completion = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                content = completion.choices[0].message.parsed
            else:
                # Regular completion
                completion = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = completion.choices[0].message.content

            usage_metadata = {
                "tokens_total": completion.usage.total_tokens,
                "tokens_input": completion.usage.prompt_tokens,
                "tokens_output": completion.usage.completion_tokens,
                "completion_id": completion.id,
                "model": model,
            }

            logger.info(
                "chat_completion_created",
                completion_id=completion.id,
                tokens_total=usage_metadata["tokens_total"],
            )

            return content, usage_metadata

        except OpenAIError as e:
            logger.error("openai_error", error=str(e), model=model, exc_info=True)
            raise

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

            response = await self.client.embeddings.create(**kwargs)

            # Extract embedding vector
            embedding = response.data[0].embedding

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

        all_embeddings = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                kwargs = {"model": model, "input": batch}
                if dimensions:
                    kwargs["dimensions"] = dimensions

                response = await self.client.embeddings.create(**kwargs)

                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Track tokens
                total_tokens += response.usage.total_tokens

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
