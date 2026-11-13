"""Process budgets for optimization proposals, retaining checked checkpoints."""

import contextlib
import io
import math
import os
import pickle
import signal
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetExhausted:
    """An optimization query stopped before producing usable proof evidence."""

    reason: str


def emit_checkpoint(value):
    """Send an immutable intermediate result when running inside a budget worker."""
    if os.environ.get("SYMBOPT_OPTIMIZATION_WORKER") == "1":
        _emit("checkpoint", value)


def _emit(kind, value):
    from funcprops import pack_analysis_data

    stream = sys.__stdout__.buffer
    stream.write(pack_analysis_data((kind, value)))
    stream.flush()


def _messages(data):
    stream = io.BytesIO(data)
    result = []
    while stream.tell() < len(data):
        try:
            result.append(pickle.load(stream))
        except (EOFError, pickle.UnpicklingError):
            break
    return result


def bounded_optimization(operation, expression, args, kwargs, seconds):
    """Run a known optimization operation with source-preserving worker transport."""
    from funcprops import pack_analysis_data

    if isinstance(seconds, bool):
        raise TypeError("work budget must be finite and positive")
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("work budget must be finite and positive")
    if operation not in {"certify_extrema_bounds", "certify_global_minimum"}:
        raise ValueError("unknown optimization operation")
    env = {**os.environ, "SYMBOPT_OPTIMIZATION_WORKER": "1"}
    timed_out = False
    with subprocess.Popen(
        [sys.executable, "-m", "symbopt.optimization_budget"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            output, _ = process.communicate(
                pack_analysis_data((operation, expression, args, kwargs)),
                timeout=seconds,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, _ = process.communicate()
        messages = _messages(output)
        if process.returncode == 0 and messages:
            kind, value = messages[-1]
            if kind == "error":
                raise value
            if kind == "return":
                return value
        checkpoints = [value for kind, value in messages if kind == "checkpoint"]
        if checkpoints:
            return checkpoints[-1]
        return BudgetExhausted(
            "work-budget-exhausted" if timed_out else "worker-failed"
        )


def _main():
    operation, expression, args, kwargs = pickle.loads(sys.stdin.buffer.read())
    from .extrema_bounds import certify_extrema_bounds
    from .univariate_certificates import certify_global_minimum

    functions = {
        "certify_extrema_bounds": certify_extrema_bounds,
        "certify_global_minimum": certify_global_minimum,
    }
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = functions[operation](expression, *args, **kwargs)
        _emit("return", result)
    except (
        TypeError,
        ValueError,
        ArithmeticError,
        NotImplementedError,
        AttributeError,
    ) as error:
        _emit("error", error)


if __name__ == "__main__":
    _main()
