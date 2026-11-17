"""Негативные контроли ЛР2.

1. Невалидная конфигурация (config_negative_test.log).
2. Ловушка утечки: масштабирование по train и test меняет предсказания.
3. Воспроизводимость: повтор с тем же зерном даёт побайтово те же предсказания.
4. Пик памяти чанкового режима не растёт с объёмом данных.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from energy.config import PipelineConfig
from energy.data import load_data, split_by_shape
from energy.model import IncrementalRidge, build_pipeline

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "lab2"
DATA_XLSX = ROOT / "data" / "ENB2012_data.xlsx"


def check_invalid_config() -> bool:
    cases = [
        ("нет файла данных", dict(data_path=ROOT / "data" / "missing.xlsx")),
        ("chunk_size = 0", dict(data_path=DATA_XLSX, chunk_size=0)),
        (
            "val_size + test_size >= 1",
            dict(data_path=DATA_XLSX, val_size=0.6, test_size=0.6),
        ),
    ]
    lines = []
    ok = True
    for label, kwargs in cases:
        try:
            PipelineConfig(**kwargs)
            ok = False
            lines.append(f"[FAIL] принята невалидная конфигурация: {label}")
        except Exception as exc:
            lines.append(f"[OK] отказ: {label}\n     {type(exc).__name__}: {exc}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "config_negative_test.log").write_text(
        "\n\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n\n".join(lines))
    return ok


def check_leakage(data: dict, alpha: float) -> tuple[bool, float]:
    Xtr = data["X_train"].to_numpy()
    Xte = data["X_test"].to_numpy()
    ytr = data["y_train"].to_numpy()

    correct = build_pipeline(alpha)
    correct.fit(data["X_train"], data["y_train"])
    pred_correct = correct.predict(data["X_test"])

    scaler_leak = StandardScaler().fit(np.vstack([Xtr, Xte]))
    leak = MultiOutputRegressor(Ridge(alpha=alpha))
    leak.fit(scaler_leak.transform(Xtr), ytr)
    pred_leak = leak.predict(scaler_leak.transform(Xte))

    diff = float(np.max(np.abs(pred_correct - pred_leak)))
    leaked = not np.allclose(pred_correct, pred_leak, rtol=1e-6, atol=1e-6)
    print(f"утечка: max |Δпредсказания| = {diff:.6f}; утечка обнаружена: {leaked}")
    return leaked, diff


def check_reproducible(cfg: PipelineConfig, alpha: float) -> tuple[bool, float]:
    df = load_data(cfg.data_path)
    d1 = split_by_shape(df, cfg.val_size, cfg.test_size, cfg.seed)
    d2 = split_by_shape(df, cfg.val_size, cfg.test_size, cfg.seed)
    split_ok = all(
        d1[f"X_{p}"].index.equals(d2[f"X_{p}"].index) for p in ("train", "val", "test")
    )

    p1 = build_pipeline(alpha).fit(d1["X_train"], d1["y_train"])
    p2 = build_pipeline(alpha).fit(d2["X_train"], d2["y_train"])
    pred1 = p1.predict(d1["X_test"])
    pred2 = p2.predict(d2["X_test"])
    diff = float(np.max(np.abs(pred1 - pred2)))
    pred_ok = np.array_equal(pred1, pred2)
    print(f"воспроизводимость: split={split_ok}, max |Δпредсказания| = {diff:.3g}")
    return split_ok and pred_ok, diff


def check_chunk_matches_full(
    cfg: PipelineConfig, data: dict, alpha: float
) -> tuple[bool, float]:
    """Чанковый и полный режимы дают один результат (расхождение отсутствует)."""
    from energy.run import fit_chunks, fit_full

    pipe, _, _ = fit_full(cfg, data, alpha)
    scaler, model, _, _ = fit_chunks(cfg, data, alpha)
    pred_full = pipe.predict(data["X_test"])
    pred_chunk = model.predict(scaler.transform(data["X_test"].to_numpy()))
    diff = float(np.max(np.abs(pred_full - pred_chunk)))
    ok = np.allclose(pred_full, pred_chunk, rtol=1e-8, atol=1e-8)
    print(f"чанки vs полный: max |Δпредсказания| = {diff:.3g}; совпадают: {ok}")
    return ok, diff


def _fit_chunks_peak(
    X: np.ndarray, y: np.ndarray, chunk_size: int, alpha: float
) -> int:
    scaler = StandardScaler()
    model = IncrementalRidge(alpha=alpha)
    n = len(X)
    tracemalloc.start()
    for i in range(0, n, chunk_size):
        scaler.partial_fit(X[i : i + chunk_size])
    for i in range(0, n, chunk_size):
        model.partial_fit(
            scaler.transform(X[i : i + chunk_size]), y[i : i + chunk_size]
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def check_memory_flat(cfg: PipelineConfig, alpha: float) -> tuple[bool, list]:
    rng = np.random.default_rng(0)
    chunk_peaks = []
    full_peaks = []
    for n in (10_000, 50_000, 200_000):
        X = rng.normal(size=(n, 8))
        y = rng.normal(size=(n, 2))

        chunk_peaks.append(_fit_chunks_peak(X, y, cfg.chunk_size, alpha))

        tracemalloc.start()
        pipe = build_pipeline(alpha)
        pipe.fit(X, y)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        full_peaks.append(p)

    print(f"пик памяти чанков по объёмам: {chunk_peaks}")
    print(f"пик памяти полного режима по объёмам: {full_peaks}")

    chunk_growth = max(chunk_peaks[1:]) - min(chunk_peaks)
    chunk_flat = chunk_growth <= max(chunk_peaks) * 0.5
    full_grows = full_peaks[-1] > full_peaks[0] * 2
    print(
        f"чанковый пик растёт с объёмом: {not chunk_flat}; полный растёт: {full_grows}"
    )
    return chunk_flat and full_grows, chunk_peaks


def main() -> int:
    cfg = PipelineConfig(data_path=DATA_XLSX)
    df = load_data(cfg.data_path)
    data = split_by_shape(df, cfg.val_size, cfg.test_size, cfg.seed)
    alpha = 1.0

    checks = {
        "invalid_config": check_invalid_config(),
        "leakage_detected": check_leakage(data, alpha)[0],
        "reproducible": check_reproducible(cfg, alpha)[0],
        "chunk_matches_full": check_chunk_matches_full(cfg, data, alpha)[0],
        "chunk_memory_flat": check_memory_flat(cfg, alpha)[0],
    }
    print("\nИТОГ:", checks)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
