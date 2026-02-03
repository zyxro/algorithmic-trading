from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaLabelConfig:
    objective: str = "binary"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


def fit_meta_label_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: MetaLabelConfig | None = None,
) -> lgb.LGBMClassifier:
    cfg = cfg or MetaLabelConfig()

    model = lgb.LGBMClassifier(
        objective=cfg.objective,
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_meta_probs(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
