"""Base agent class for all RSAS agents using OpenAI Response API."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel

from ..models.audit import AgentTrace
from ..models.base import AgentContext, AgentResult
from ..models.enums import AgentType
from ...integrations.openai_client import OpenAIClient
from ...observability.logger import get_logger

logger = get_logger(__name__)

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Abstract base class for all agents.

    Implements template method pattern with:
    - Idempotency checking via input hash
    - Full trace storage
    - Automatic retry via OpenAI client
    - Structured input/output with Pydantic
    - Response API integration
    """

    def __init__(self, store=None):
        """Initialize base agent.

        Args:
            store: ObjectStore instance for persistence
        """
        self.store = store
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the agent type enum.

        Must be implemented by subclass.
        """
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[TOutput]:
        """Return the Pydantic schema for output.

        Must be implemented by subclass.
        """
        pass

    async def execute(
        self, input_data: TInput, context: AgentContext
    ) -> AgentResult[TOutput]:
        """Main execution method with idempotency, trace, and retry.

        This is the template method that orchestrates the execution flow.

        Args:
            input_data: Input data (Pydantic model)
            context: Execution context

        Returns:
            AgentResult with success status and data

        Raises:
            Exception: If execution fails after retries
        """
        start_time = time.time()

        self.logger.info(
            "agent_execution_start",
            agent=self.agent_type.value,
            job_id=context.job_id,
            trace_id=context.trace_id,
        )

        try:
            # 1. Validate input
            if not await self.validate_input(input_data, context):
                return AgentResult(
                    success=False,
                    data=None,
                    error="Input validation failed",
                )

            # 2. Check for cached result (idempotency)
            input_hash = self._hash_input(input_data)

            if context.config.get("pipeline", {}).get("idempotency", True):
                cached_result = await self._get_cached_trace(context.job_id, input_hash)
                if cached_result:
                    self.logger.info(
                        "using_cached_result",
                        agent=self.agent_type.value,
                        job_id=context.job_id,
                        input_hash=input_hash,
                    )
                    return AgentResult(
                        success=True,
                        data=self.output_schema(**cached_result["output_data"]),
                        tokens_used=cached_result.get("tokens_used", 0),
                        duration_ms=cached_result.get("duration_ms", 0),
                        metadata={"from_cache": True},
                    )

            # 3. Process (call Response API)
            result = await self.process(input_data, context)

            # 4. Validate output
            if result.success and result.data:
                if not await self.validate_output(result.data, context):
                    return AgentResult(
                        success=False,
                        data=None,
                        error="Output validation failed",
                    )

            # 5. Store trace
            duration_ms = int((time.time() - start_time) * 1000)
            await self._store_trace(
                context.job_id,
                input_hash,
                result,
                duration_ms,
                context,
            )

            self.logger.info(
                "agent_execution_complete",
                agent=self.agent_type.value,
                job_id=context.job_id,
                success=result.success,
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.error(
                "agent_execution_failed",
                agent=self.agent_type.value,
                job_id=context.job_id,
                error=str(e),
                duration_ms=duration_ms,
                exc_info=True,
            )

            # Store failed trace
            await self._store_trace(
                context.job_id,
                self._hash_input(input_data),
                AgentResult(success=False, data=None, error=str(e)),
                duration_ms,
                context,
            )

            raise

    @abstractmethod
    async def process(
        self, input_data: TInput, context: AgentContext
    ) -> AgentResult[TOutput]:
        """Process the input and return result.

        Must be implemented by subclass. This is where the agent-specific
        logic goes, including calling the OpenAI Response API.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            AgentResult with data
        """
        pass

    @abstractmethod
    def _build_prompt(self, input_data: TInput, context: AgentContext) -> str:
        """Build prompt for LLM.

        Must be implemented by subclass.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            Prompt string
        """
        pass

    async def validate_input(self, input_data: TInput, context: AgentContext) -> bool:
        """Validate input before processing.

        Can be overridden by subclass for custom validation.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            True if valid
        """
        return True

    async def validate_output(self, output_data: TOutput, context: AgentContext) -> bool:
        """Validate output after processing.

        Can be overridden by subclass for custom validation.

        Args:
            output_data: Output data
            context: Execution context

        Returns:
            True if valid
        """
        return True

    async def _call_response_api(
        self, prompt: str, context: AgentContext
    ) -> tuple[TOutput, dict[str, Any]]:
        """Call OpenAI Response API with structured output.

        Args:
            prompt: Prompt text
            context: Execution context

        Returns:
            Tuple of (parsed output, metadata)
        """
        # Create OpenAI client from context
        from ...integrations.openai_client import OpenAIClient
        import os

        api_key = os.getenv(context.config.get("openai", {}).get("api_key_env", "OPENAI_API_KEY"))
        client = OpenAIClient(api_key=api_key)

        # Get model and reasoning effort from config
        model = context.config.get("openai", {}).get("model", "gpt-5.1")
        reasoning_effort = context.config.get("openai", {}).get("reasoning_effort", "high")

        # Call Response API
        output, metadata = await client.create_response(
            input_text=prompt,
            response_model=self.output_schema,
            model=model,
            reasoning_effort=reasoning_effort,
            metadata={
                "job_id": context.job_id,
                "agent": self.agent_type.value,
                "trace_id": context.trace_id,
            },
        )

        return output, metadata

    def _hash_input(self, input_data: TInput) -> str:
        """Create SHA-256 hash of input for idempotency.

        Args:
            input_data: Input data

        Returns:
            Hexadecimal hash string
        """
        # Convert to JSON and hash
        import json
        data_dict = input_data.model_dump(mode='json')
        json_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _get_cached_trace(
        self, job_id: str, input_hash: str
    ) -> dict[str, Any] | None:
        """Get cached agent trace for idempotency.

        Args:
            job_id: Job identifier
            input_hash: Input hash

        Returns:
            Trace data if exists, None otherwise
        """
        if not self.store:
            return None

        trace = self.store.get_agent_trace(job_id, self.agent_type.value, input_hash)
        if trace and trace.get("success"):
            return {
                "output_data": trace.get("output_data"),
                "tokens_used": trace.get("tokens_used", 0),
                "duration_ms": trace.get("duration_ms", 0),
            }
        return None

    async def _store_trace(
        self,
        job_id: str,
        input_hash: str,
        result: AgentResult[TOutput],
        duration_ms: int,
        context: AgentContext,
    ) -> None:
        """Store agent execution trace.

        Args:
            job_id: Job identifier
            input_hash: Input hash
            result: Agent result
            duration_ms: Execution duration
            context: Agent context for config
        """
        if not self.store:
            return

        trace = AgentTrace(
            job_id=job_id,
            agent_type=self.agent_type,
            input_hash=input_hash,
            output_data=result.data.model_dump(mode='json') if result.data else None,
            success=result.success,
            error=result.error,
            duration_ms=duration_ms,
            tokens_used=result.tokens_used,
            tokens_input=result.metadata.get("tokens_input", 0),
            tokens_output=result.metadata.get("tokens_output", 0),
            model_used=context.config.get("openai", {}).get("model"),
            reasoning_effort=context.config.get("openai", {}).get("reasoning_effort"),
            metadata=result.metadata,
        )

        self.store.save_agent_trace(job_id, self.agent_type.value, input_hash, trace)
