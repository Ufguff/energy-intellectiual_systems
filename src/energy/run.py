"""Запуск ЛР2: ML-пайплайн на полном датасете и на чанках."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from energy.config import PipelineConfig
from energy.data import TARGETS, load_data, split_by_shape
from energy.metrics import METRICS, bootstrap
from energy.model import IncrementalRidge, build_pipeline

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "lab2"
DATA_XLSX = ROOT / "data" / "ENB2012_data.xlsx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def tune_alpha(cfg: PipelineConfig, data: dict) -> float:
    """Подбор alpha только на проверочной части."""
    best_alpha, best_score = cfg.alpha_grid[0], float("inf")
    for a in cfg.alpha_grid:
        pipe = build_pipeline(a)
        pipe.fit(data["X_train"], data["y_train"])
        pred = pipe.predict(data["X_val"])
        score = float(np.mean(METRICS["rmse"](data["y_val"].to_numpy(), pred)))
        if score < best_score:
            best_score, best_alpha = score, a
    return best_alpha


def fit_full(cfg: PipelineConfig, data: dict, alpha: float):
    pipe = build_pipeline(alpha)
    tracemalloc.start()
    t0 = time.perf_counter()
    pipe.fit(data["X_train"], data["y_train"])
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return pipe, elapsed, peak


def fit_chunks(cfg: PipelineConfig, data: dict, alpha: float):
    scaler = StandardScaler()
    model = IncrementalRidge(alpha=alpha)
    X, y = data["X_train"], data["y_train"]
    n = len(X)
    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(0, n, cfg.chunk_size):
        scaler.partial_fit(X.iloc[i : i + cfg.chunk_size].to_numpy())
    for i in range(0, n, cfg.chunk_size):
        Xc = scaler.transform(X.iloc[i : i + cfg.chunk_size].to_numpy())
        model.partial_fit(Xc, y.iloc[i : i + cfg.chunk_size].to_numpy())
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return scaler, model, elapsed, peak


def eval_metrics(y_true, y_pred, cfg: PipelineConfig) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for name, fn in METRICS.items():
        med, lo, hi = bootstrap(y_true, y_pred, fn, cfg.n_bootstrap, cfg.seed)
        out[name] = {"value": med, "lo": lo, "hi": hi}
    return out


def naive_baseline(data: dict) -> dict:
    base = DummyRegressor(strategy="mean")
    base.fit(data["X_train"], data["y_train"])
    pred = base.predict(data["X_test"])
    y = data["y_test"].to_numpy()
    return {name: fn(y, pred) for name, fn in METRICS.items()}


def log_run(
    mode: str,
    cfg: PipelineConfig,
    alpha: float,
    metrics: dict,
    elapsed: float,
    peak: int,
) -> str:
    flat = {}
    for metric, m in metrics.items():
        for ti, target in enumerate(TARGETS):
            short = "Y1" if target == "heating_load" else "Y2"
            flat[f"{short}_{metric}"] = float(m["value"][ti])
            flat[f"{short}_{metric}_lo"] = float(m["lo"][ti])
            flat[f"{short}_{metric}_hi"] = float(m["hi"][ti])
    flat["train_time_s"] = elapsed
    flat["peak_mem_bytes"] = peak
    with mlflow.start_run(run_name=f"lab2_{mode}") as run:
        mlflow.log_params(
            {
                "model": cfg.model,
                "alpha": alpha,
                "chunk_size": cfg.chunk_size,
                "seed": cfg.seed,
                "mode": mode,
            }
        )
        mlflow.log_metrics(flat)
        return run.info.run_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ЛР2: ML-пайплайн, полный датасет vs чанки"
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.config is not None:
        cfg = PipelineConfig.model_validate_json(
            args.config.read_text(encoding="utf-8")
        )
    else:
        cfg = PipelineConfig(data_path=DATA_XLSX)

    REPORTS.mkdir(parents=True, exist_ok=True)
    df = load_data(cfg.data_path)
    data = split_by_shape(df, cfg.val_size, cfg.test_size, cfg.seed)

    best_alpha = tune_alpha(cfg, data)

    # Полный датасет.
    pipe, t_full, mem_full = fit_full(cfg, data, best_alpha)
    metrics_full = eval_metrics(
        data["y_test"].to_numpy(), pipe.predict(data["X_test"]), cfg
    )

    # Чанки.
    scaler, inc_model, t_chunk, mem_chunk = fit_chunks(cfg, data, best_alpha)
    X_test_s = scaler.transform(data["X_test"].to_numpy())
    metrics_chunk = eval_metrics(
        data["y_test"].to_numpy(), inc_model.predict(X_test_s), cfg
    )

    naive = naive_baseline(data)

    # MLflow: локальный sqlite-бэкенд.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    run_full = log_run("full", cfg, best_alpha, metrics_full, t_full, mem_full)
    run_chunk = log_run("chunks", cfg, best_alpha, metrics_chunk, t_chunk, mem_chunk)

    # Модель с хешем.
    model_path = REPORTS / "ml_model.joblib"
    joblib.dump(pipe, model_path)
    model_hash = sha256(model_path)

    # ml_metrics.csv.
    rows = []
    for mode, metrics, run_id in [
        ("full", metrics_full, run_full),
        ("chunks", metrics_chunk, run_chunk),
    ]:
        for metric in ["mae", "rmse", "r2"]:
            for ti, target in enumerate(TARGETS):
                rows.append(
                    {
                        "mode": mode,
                        "target": target,
                        "metric": metric,
                        "value": metrics[metric]["value"][ti],
                        "lo": metrics[metric]["lo"][ti],
                        "hi": metrics[metric]["hi"][ti],
                        "run_id": run_id,
                    }
                )
    pd.DataFrame(rows).to_csv(REPORTS / "ml_metrics.csv", index=False)

    results = {
        "config": cfg.model_dump(mode="json"),
        "best_alpha": best_alpha,
        "split": data["shapes"],
        "train_size": int(len(data["X_train"])),
        "val_size": int(len(data["X_val"])),
        "test_size": int(len(data["X_test"])),
        "model_sha256": model_hash,
        "run_full": run_full,
        "run_chunk": run_chunk,
        "full": {
            "time_s": t_full,
            "peak_mem_bytes": mem_full,
            "metrics": _to_plain(metrics_full),
        },
        "chunks": {
            "time_s": t_chunk,
            "peak_mem_bytes": mem_chunk,
            "metrics": _to_plain(metrics_chunk),
        },
        "naive_baseline": _to_plain(naive),
    }
    (REPORTS / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
