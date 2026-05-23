import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterator, Optional

from config import LOG_DIR


SERVICE_PERF_LOG_PATH = os.path.join(LOG_DIR, "service_perf.log")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def log_event(message: str, **fields) -> None:
    _ensure_log_dir()
    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        **fields,
    }
    line = json.dumps(item, ensure_ascii=False)
    print(f"[rag-service] {line}", file=sys.stderr, flush=True)
    with open(SERVICE_PERF_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@contextmanager
def timed_stage(name: str, **fields) -> Iterator[Dict[str, Optional[float]]]:
    start = time.perf_counter()
    log_event(f"{name} started", **fields)
    state: Dict[str, Optional[float]] = {"cost_seconds": None}
    try:
        yield state
    finally:
        cost = time.perf_counter() - start
        state["cost_seconds"] = cost
        log_event(f"{name} finished", cost_seconds=round(cost, 4), **fields)


def env_snapshot() -> Dict[str, str]:
    return {
        "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.getenv("TRANSFORMERS_OFFLINE", ""),
        "HF_HOME": os.getenv("HF_HOME", ""),
        "SENTENCE_TRANSFORMERS_HOME": os.getenv("SENTENCE_TRANSFORMERS_HOME", ""),
    }
