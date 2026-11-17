"""Схема настроек пайплайна на pydantic.

Невалидная конфигурация (несуществующий путь, нулевой размер чанка,
недопустимое разбиение) приводит к отказу ещё до запуска обучения.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineConfig(BaseModel):
    data_path: Path
    seed: int = 42
    chunk_size: int = 128
    model: Literal["ridge"] = "ridge"
    alpha: float = 1.0
    n_bootstrap: int = 2000
    val_size: float = 0.15
    test_size: float = 0.15
    alpha_grid: list[float] = Field(
        default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    )

    @field_validator("data_path")
    @classmethod
    def _path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"data_path не существует: {v}")
        return v

    @field_validator("chunk_size", "n_bootstrap")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"значение должно быть > 0, получено {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def _positive_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"alpha должно быть > 0, получено {v}")
        return v

    @model_validator(mode="after")
    def _check_splits(self) -> "PipelineConfig":
        if not (0 < self.val_size < 1 and 0 < self.test_size < 1):
            raise ValueError("val_size и test_size должны лежать в (0, 1)")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size должно быть < 1")
        return self
