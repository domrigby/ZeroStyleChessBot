import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, List

import psutil
import pytest

from tests.integration.tiny_pipeline import (
    NUM_AGENTS,
    NUM_EVALUATORS,
    NUM_TRAINERS,
    SHUTDOWN_TIMEOUT_SECONDS,
    FaultyTrainer,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "tests" / "integration" / "tiny_pipeline.py"
NUM_PIPELINE_CHILDREN = NUM_AGENTS + NUM_EVALUATORS + NUM_TRAINERS
STARTUP_TIMEOUT_SECONDS = 60.0
EXIT_GRACE_SECONDS = 15.0


def _descendants(process: subprocess.Popen) -> List[psutil.Process]:
    return psutil.Process(process.pid).children(recursive=True)


def _wait_for_children(process: subprocess.Popen) -> List[psutil.Process]:
    """Wait until every pipeline child has been spawned (plus multiprocessing's resource tracker)."""
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        assert process.poll() is None, "pipeline exited during startup"
        descendants = _descendants(process)
        if len(descendants) > NUM_PIPELINE_CHILDREN:
            return descendants
        time.sleep(0.5)
    raise TimeoutError("pipeline children did not start")


def _assert_no_orphans(descendants: List[psutil.Process]) -> None:
    _, still_alive = psutil.wait_procs(descendants, timeout=EXIT_GRACE_SECONDS)
    assert still_alive == []


@pytest.fixture
def launch_pipeline(tmp_path: Path) -> Iterator[Callable[..., subprocess.Popen]]:
    """Launch tests/integration/tiny_pipeline.py with tmp_path as both run directory and working directory."""
    launched: List[subprocess.Popen] = []

    def launch(*extra_args: str) -> subprocess.Popen:
        environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        with open(tmp_path / "pipeline_output.txt", "w") as output:
            process = subprocess.Popen(
                [sys.executable, str(LAUNCHER), "--run_dir", str(tmp_path), *extra_args],
                cwd=tmp_path,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        launched.append(process)
        return process

    yield launch

    for process in launched:
        if process.poll() is None:
            for child in _descendants(process):
                child.kill()
            process.kill()
            process.wait()


def test_sigint_stops_all_children_and_saves_checkpoint(launch_pipeline, tmp_path: Path) -> None:
    run_seconds = 20.0
    process = launch_pipeline()
    descendants = _wait_for_children(process)

    time.sleep(run_seconds)
    assert process.poll() is None, "pipeline died before SIGINT"
    process.send_signal(signal.SIGINT)

    assert process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS + EXIT_GRACE_SECONDS) == 0
    assert list(tmp_path.glob("RL_tuned_*.pt")) != []
    _assert_no_orphans(descendants)


def test_injected_trainer_fault_shuts_down_with_non_zero_exit(launch_pipeline, tmp_path: Path) -> None:
    process = launch_pipeline("--inject_fault")
    descendants = _wait_for_children(process)

    assert process.wait(timeout=STARTUP_TIMEOUT_SECONDS + SHUTDOWN_TIMEOUT_SECONDS) != 0
    assert "Injected trainer fault" in (tmp_path / f"{FaultyTrainer.__name__}.txt").read_text()
    _assert_no_orphans(descendants)


@pytest.mark.slow
def test_resident_memory_stays_bounded_over_two_minutes(launch_pipeline) -> None:
    run_seconds = 120.0
    warmup_seconds = 30.0
    sample_period_seconds = 5.0
    allowed_growth_factor = 1.25

    process = launch_pipeline()
    _wait_for_children(process)

    start_time = time.time()
    samples: List[tuple] = []
    while time.time() - start_time < run_seconds:
        assert process.poll() is None, "pipeline died during the run"
        tree = [psutil.Process(process.pid), *_descendants(process)]
        total_rss = 0
        for member in tree:
            try:
                total_rss += member.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        samples.append((time.time() - start_time, total_rss))
        time.sleep(sample_period_seconds)

    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS + EXIT_GRACE_SECONDS) == 0

    after_warmup = [rss for elapsed, rss in samples if elapsed >= warmup_seconds]
    print(f"Total RSS samples (s, MB): {[(round(t), rss // 2**20) for t, rss in samples]}")
    assert max(after_warmup) <= allowed_growth_factor * after_warmup[0]
