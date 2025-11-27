"""Cost tracking utilities for budget monitoring and analytics (file-based)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ...core.storage.object_store import ObjectStore
from ...observability.logger import get_logger

logger = get_logger(__name__)


class CostTracker:
    """Track and monitor API costs for budget control using the object store."""

    def __init__(self, store: ObjectStore):
        self.store = store
        self.log_path = Path(self.store.base_dir) / "costs.json"
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict[str, Any]]:
        import json

        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, data: list[dict[str, Any]]) -> None:
        import json

        self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def record_cost(
        self,
        operation_type: str,
        tokens_used: int,
        cost_usd: float,
        model_used: str | None = None,
        search_query_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry = {
            "id": f"cost_{len(self._read()) + 1}",
            "operation_type": operation_type,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "model_used": model_used,
            "search_query_id": search_query_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows = self._read()
        rows.append(entry)
        self._write(rows)
        self.logger.info("cost_recorded", operation=operation_type, cost=cost_usd, tokens=tokens_used)
        return entry["id"]

    async def get_total_cost(
        self,
        operation_type: str | None = None,
        since: datetime | None = None,
    ) -> float:
        rows = self._read()
        total = 0.0
        for row in rows:
            if operation_type and row.get("operation_type") != operation_type:
                continue
            created = row.get("created_at")
            if since and created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt < since:
                        continue
                except Exception:
                    pass
            total += float(row.get("cost_usd", 0.0))
        return total

    async def get_daily_cost(self) -> float:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.get_total_cost(since=today)

    async def get_monthly_cost(self) -> float:
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return await self.get_total_cost(since=month_start)

    async def get_cost_breakdown(self, since: datetime | None = None) -> dict[str, float]:
        rows = self._read()
        breakdown: dict[str, float] = {}
        for row in rows:
            created = row.get("created_at")
            if since and created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt < since:
                        continue
                except Exception:
                    pass
            op = row.get("operation_type")
            breakdown[op] = breakdown.get(op, 0.0) + float(row.get("cost_usd", 0.0))
        self.logger.info("cost_breakdown_retrieved", operations=len(breakdown))
        return breakdown

    async def check_budget_limit(
        self,
        daily_limit: float | None = None,
        monthly_limit: float | None = None,
    ) -> dict[str, Any]:
        daily_cost = await self.get_daily_cost()
        monthly_cost = await self.get_monthly_cost()

        daily_exceeded = daily_limit is not None and daily_cost >= daily_limit
        monthly_exceeded = monthly_limit is not None and monthly_cost >= monthly_limit

        status = {
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "daily_exceeded": daily_exceeded,
            "monthly_exceeded": monthly_exceeded,
            "any_exceeded": daily_exceeded or monthly_exceeded,
        }

        if status["any_exceeded"]:
            self.logger.warning(
                "budget_limit_exceeded",
                daily_cost=daily_cost,
                monthly_cost=monthly_cost,
                daily_limit=daily_limit,
                monthly_limit=monthly_limit,
            )

        return status


def estimate_operation_cost(
    operation_type: str,
    tokens: int | None = None,
    candidates: int | None = None,
) -> float:
    """Estimate cost for an operation."""
    costs = {
        "summary": 0.001,
        "embedding": 0.000006,
        "tier3_analysis": 0.03,
        "tier3_compare": 0.30,
    }

    if operation_type in costs:
        base_cost = costs[operation_type]
        if candidates:
            return base_cost * candidates
        return base_cost

    if tokens:
        return (tokens / 1_000_000) * 0.40

    return 0.0
