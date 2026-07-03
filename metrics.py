import logging
import time
from contextlib import contextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineMetrics:
    def __init__(self):
        self._timings = defaultdict(list)

    @contextmanager
    def timing(self, stage: str):
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self._timings[stage].append(elapsed)
            logger.debug("[Timing] %s: %.3fс", stage, elapsed)

    def get_stats(self) -> dict[str, dict]:
        stats = {}
        for stage, times in self._timings.items():
            stats[stage] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
            }
        return stats

    def log_summary(self):
        stats = self.get_stats()
        if not stats:
            return
        parts = []
        for stage, s in stats.items():
            parts.append(f"{stage}={s['avg']:.2f}с")
        logger.info("Пайплайна статистика: %s", ", ".join(parts))

    def reset(self):
        self._timings.clear()


metrics = PipelineMetrics()
