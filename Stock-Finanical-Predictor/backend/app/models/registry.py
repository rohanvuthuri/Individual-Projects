"""Keeps trained models in memory and on disk, and trains the LSTM in the background.

Gradient boosting trains in a few seconds, so it trains during the request.
The LSTM can take a minute, so the first request starts a background job and
reports "training"; the frontend polls until it is ready.
"""

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ..modes import Mode
from . import gbm, lstm, stacking

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "model_cache"
ERROR_RETRY_SECONDS = 60
# Bump when saved bundles change shape so old files on disk get retrained.
# v2: out-of-fold predictions + stacking meta-model.
BUNDLE_VERSION = 2

_memory: dict[tuple[str, str, str], dict] = {}
_key_locks: dict[tuple[str, str, str], threading.Lock] = {}
_locks_guard = threading.Lock()

# One worker: TensorFlow already uses every CPU core for a single model.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lstm")
_jobs: dict[tuple[str, str], Future] = {}
_errors: dict[tuple[str, str], tuple[float, str]] = {}
_jobs_guard = threading.Lock()


def _key_lock(key: tuple[str, str, str]) -> threading.Lock:
    with _locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _path(ticker: str, mode: Mode, kind: str, suffix: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]", "_", ticker)
    return CACHE_DIR / f"{safe}_{mode.key}_{kind}{suffix}"


def _is_fresh(bundle: dict | None, mode: Mode) -> bool:
    return (
        bundle is not None
        and bundle.get("version") == BUNDLE_VERSION
        and datetime.now(timezone.utc) - bundle["trained_at"] < mode.model_max_age
    )


# ---------------------------------------------------------------- gradient boosting

def _load_gbm(ticker: str, mode: Mode) -> dict | None:
    key = (ticker, mode.key, "gbm")
    if key in _memory:
        return _memory[key]
    path = _path(ticker, mode, "gbm", ".joblib")
    if path.exists():
        try:
            _memory[key] = joblib.load(path)
            return _memory[key]
        except Exception:
            log.warning("Could not load %s, retraining", path)
    return None


def get_gbm(ticker: str, mode: Mode, X: pd.DataFrame, fwd_ret: pd.Series, close: pd.Series) -> dict:
    key = (ticker, mode.key, "gbm")
    with _key_lock(key):
        bundle = _load_gbm(ticker, mode)
        if _is_fresh(bundle, mode) and bundle["features"] == list(X.columns):
            return bundle
        bundle = gbm.train(X, fwd_ret, close, mode)
        bundle["version"] = BUNDLE_VERSION
        CACHE_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, _path(ticker, mode, "gbm", ".joblib"))
        _memory[key] = bundle
        return bundle


# ---------------------------------------------------------------- LSTM

def _load_lstm(ticker: str, mode: Mode) -> dict | None:
    key = (ticker, mode.key, "lstm")
    if key in _memory:
        return _memory[key]
    model_path = _path(ticker, mode, "lstm", ".keras")
    meta_path = _path(ticker, mode, "lstm", ".joblib")
    if model_path.exists() and meta_path.exists():
        try:
            from keras.models import load_model

            bundle = joblib.load(meta_path)
            bundle["model"] = load_model(model_path)
            _memory[key] = bundle
            return bundle
        except Exception:
            log.warning("Could not load %s, retraining", model_path)
    return None


def _train_lstm_job(ticker: str, mode: Mode, X: pd.DataFrame, fwd_ret: pd.Series) -> None:
    bundle = lstm.train(X, fwd_ret, mode)
    bundle["version"] = BUNDLE_VERSION

    # The meta-model needs both models' out-of-fold predictions, so it trains
    # once the LSTM's are ready. It rides along in the LSTM's saved bundle.
    g = _load_gbm(ticker, mode)
    bundle["stack"] = None
    if g is not None and "oof_prob" in g:
        try:
            bundle["stack"] = stacking.train(g["oof_prob"], bundle["oof_pred"], X, fwd_ret, mode)
        except Exception:
            log.exception("Stacking meta-model failed for %s %s; falling back to the agreement rule", ticker, mode.key)

    CACHE_DIR.mkdir(exist_ok=True)
    bundle["model"].save(_path(ticker, mode, "lstm", ".keras"))
    joblib.dump({k: v for k, v in bundle.items() if k != "model"}, _path(ticker, mode, "lstm", ".joblib"))
    _memory[(ticker, mode.key, "lstm")] = bundle


def _on_done(job_key: tuple[str, str], future: Future) -> None:
    with _jobs_guard:
        _jobs.pop(job_key, None)
        if future.exception() is not None:
            log.error("LSTM training failed for %s", job_key, exc_info=future.exception())
            _errors[job_key] = (time.time(), str(future.exception()))


def get_lstm(ticker: str, mode: Mode, X: pd.DataFrame, fwd_ret: pd.Series) -> tuple[str, dict | None, str | None, bool]:
    """Returns (status, bundle, error, retraining). Status is "ready", "training" or "error".

    A stale model is still returned as "ready" while a fresh one trains behind
    it; `retraining` tells the frontend to keep polling for the new one.
    """
    job_key = (ticker, mode.key)
    bundle = _load_lstm(ticker, mode)
    if bundle is not None and bundle["features"] != list(X.columns):
        bundle = None
    if _is_fresh(bundle, mode):
        return "ready", bundle, None, False

    with _jobs_guard:
        if job_key not in _jobs:
            err = _errors.get(job_key)
            if err and time.time() - err[0] < ERROR_RETRY_SECONDS:
                return ("ready", bundle, None, False) if bundle else ("error", None, err[1], False)
            _errors.pop(job_key, None)
            future = _executor.submit(_train_lstm_job, ticker, mode, X.copy(), fwd_ret.copy())
            future.add_done_callback(lambda f: _on_done(job_key, f))
            _jobs[job_key] = future

    return ("ready", bundle, None, True) if bundle else ("training", None, None, True)
