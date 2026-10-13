#!/usr/bin/env python3
"""
Validacao walk-forward dos modelos finalistas: KNN (PCA) x Random Forest (PCA).

Complementa o teste unico 70/30 do notebook model_training.ipynb (secao 7.5 do
documento). Cada mes da serie e previsto por um modelo treinado apenas com os
meses anteriores (janela crescente), como acontece em producao:

- StandardScaler + PCA(3) + modelo reajustados a cada mes, so com o treino
- mesmos hiperparametros do notebook para KNN (PCA) e Random Forest (PCA)
- limiar de decisao 0,5 sobre a probabilidade de crise
- bootstrap em blocos de 12 meses (mesmo metodo da secao 7.4)

Uso:
    python validacao_walk_forward.py
    python validacao_walk_forward.py --dataset ./raw/dataset_merged_binary.csv --n-boot 10000
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES = [
    "credit_spread",
    "credit_spread_diff",
    "industrial_production",
    "industrial_production_mom",
    "industrial_production_yoy",
    "price_index",
    "inflation_mom",
    "inflation_yoy",
    "real_interest",
    "real_interest_diff",
    "volatility_index",
    "volatility_index_mom",
    "yield_curve_slope",
    "yield_curve_slope_diff",
]
TARGET = "target_12m"
THRESHOLD = 0.5
BLOCK_SIZE = 12
SEED = 42


# ==========================
# Modelos (hiperparametros do model_training.ipynb, variantes com PCA)
# ==========================

def new_knn_pca():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            metric="manhattan",
            n_neighbors=3,
            weights="distance",
        )),
    ])


def new_rf_pca():
    return RandomForestClassifier(
        random_state=42,
        class_weight=None,
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=100,
    )


MODELS = {
    "knn": ("KNN (PCA)", new_knn_pca),
    "rf": ("Random Forest (PCA)", new_rf_pca),
}


# ==========================
# Walk-forward
# ==========================

def crisis_probability(train: pd.DataFrame, test: pd.DataFrame, new_model) -> float:
    """Ajusta scaler + PCA + modelo no treino e retorna a probabilidade de crise do mes de teste."""
    y_train = train[TARGET].astype(int)
    if 1 not in set(y_train):
        return 0.0

    scaler = StandardScaler()
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(scaler.fit_transform(train[FEATURES]))
    X_test = pca.transform(scaler.transform(test[FEATURES]))

    model = new_model()
    model.fit(X_train, y_train)
    return float(model.predict_proba(X_test)[0][list(model.classes_).index(1)])


def walk_forward(df: pd.DataFrame, min_train: int) -> pd.DataFrame:
    rows = []
    total = len(df) - min_train
    for i in range(min_train, len(df)):
        train, test = df.iloc[:i], df.iloc[[i]]
        row = {"date": df["date"].iloc[i], "y": int(df[TARGET].iloc[i])}
        for key, (_, new_model) in MODELS.items():
            probability = crisis_probability(train, test, new_model)
            row[f"prob_{key}"] = probability
            row[f"pred_{key}"] = int(probability >= THRESHOLD)
        rows.append(row)

        done = i - min_train + 1
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} meses avaliados")
    return pd.DataFrame(rows)


# ==========================
# Metricas
# ==========================

def classification_metrics(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Recall Crise, Recall Normal, Precisao macro, F1 macro."""
    return np.array([
        recall_score(y, pred, zero_division=0),
        recall_score(1 - y, 1 - pred, zero_division=0),
        precision_score(y, pred, average="macro", zero_division=0),
        f1_score(y, pred, average="macro", zero_division=0),
    ])


METRIC_NAMES = ["Recall Crise", "Recall Normal", "Precisao macro", "F1 macro"]


def period_summary(preds: pd.DataFrame, period: str) -> list[dict]:
    y = preds["y"].to_numpy()
    rows = []
    for key, (name, _) in MODELS.items():
        pred = preds[f"pred_{key}"].to_numpy()
        prob = preds[f"prob_{key}"].to_numpy()
        has_both = len(set(y)) > 1
        row = {
            "periodo": period,
            "inicio": preds["date"].min().strftime("%Y-%m"),
            "fim": preds["date"].max().strftime("%Y-%m"),
            "meses": len(preds),
            "modelo": name,
            "crises_identificadas": f"{int(((pred == 1) & (y == 1)).sum())}/{int(y.sum())}",
            "normais_identificados": f"{int(((pred == 0) & (y == 0)).sum())}/{int((y == 0).sum())}",
            **dict(zip(METRIC_NAMES, classification_metrics(y, pred).round(3))),
            "AUC": round(roc_auc_score(y, prob), 3) if has_both else np.nan,
            "Brier": round(brier_score_loss(y, prob), 3),
        }
        rows.append(row)
    return rows


def block_bootstrap(preds: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Diferenca KNN - RF em reamostras por blocos de 12 meses consecutivos."""
    rng = np.random.default_rng(SEED)
    y = preds["y"].to_numpy()
    knn = preds["pred_knn"].to_numpy()
    rf = preds["pred_rf"].to_numpy()
    n = len(preds)
    starts = np.arange(n - BLOCK_SIZE + 1)
    n_blocks = int(np.ceil(n / BLOCK_SIZE))

    diffs = np.empty((n_boot, len(METRIC_NAMES)))
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in rng.choice(starts, n_blocks)])[:n]
        diffs[b] = classification_metrics(y[idx], knn[idx]) - classification_metrics(y[idx], rf[idx])

    observed = classification_metrics(y, knn) - classification_metrics(y, rf)
    rows = []
    for j, name in enumerate(METRIC_NAMES):
        low, high = np.percentile(diffs[:, j], [2.5, 97.5])
        rows.append({
            "metrica": name,
            "diferenca_knn_menos_rf": round(observed[j], 3),
            "ic95_inferior": round(low, 3),
            "ic95_superior": round(high, 3),
            "significativa": "Sim" if low > 0 or high < 0 else "Nao",
            "pct_reamostras_rf_melhor": round(100 * (diffs[:, j] < 0).mean(), 1),
            "pct_reamostras_knn_melhor": round(100 * (diffs[:, j] > 0).mean(), 1),
        })
    return pd.DataFrame(rows)


def knn_extremes(preds: pd.DataFrame) -> pd.DataFrame:
    """Quantas vezes o KNN deu exatamente 0%/100% e quanto isso acertou."""
    prob = preds["prob_knn"]
    zero = preds[prob == 0]
    one = preds[prob == 1]
    return pd.DataFrame([{
        "meses_avaliados": len(preds),
        "meses_prob_0_ou_100": int(((prob == 0) | (prob == 1)).sum()),
        "meses_prob_0": len(zero),
        "crise_real_quando_prob_0": int(zero["y"].sum()),
        "pct_crise_real_quando_prob_0": round(100 * zero["y"].mean(), 1) if len(zero) else np.nan,
        "meses_prob_100": len(one),
        "normal_real_quando_prob_100": int((1 - one["y"]).sum()),
        "pct_normal_real_quando_prob_100": round(100 * (1 - one["y"]).mean(), 1) if len(one) else np.nan,
    }])


# ==========================
# Main
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Validacao walk-forward KNN (PCA) x Random Forest (PCA).")
    parser.add_argument("--dataset", default="./raw/dataset_merged_binary.csv", help="CSV de entrada")
    parser.add_argument("--output-dir", default="./output", help="Diretorio base de saida")
    parser.add_argument("--min-train", type=int, default=120, help="Meses minimos de historico no primeiro treino")
    parser.add_argument("--n-boot", type=int, default=10000, help="Numero de reamostras do bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.dataset, parse_dates=["date"])
    df = df[["date", *FEATURES, TARGET]].dropna().sort_values("date").reset_index(drop=True)

    # Mesmo corte do notebook: train_test_split(test_size=0.3, shuffle=False)
    train_idx, _ = train_test_split(np.arange(len(df)), test_size=0.3, shuffle=False)
    holdout_start = df["date"].iloc[len(train_idx)]

    print(f"Dataset: {len(df)} meses ({df['date'].min():%Y-%m} a {df['date'].max():%Y-%m})")
    print(f"Inicio do teste do notebook (70/30): {holdout_start:%Y-%m}")
    print(f"Walk-forward com minimo de {args.min_train} meses de treino...")

    preds = walk_forward(df, args.min_train)

    summary = pd.DataFrame(
        period_summary(preds, "walk-forward completo")
        + period_summary(preds[preds["date"] >= holdout_start], "periodo de teste do notebook")
        + period_summary(preds[preds["date"] < holdout_start], "antes do teste do notebook")
    )
    print(f"Bootstrap em blocos de {BLOCK_SIZE} meses ({args.n_boot} reamostras)...")
    bootstrap = block_bootstrap(preds, args.n_boot)
    extremes = knn_extremes(preds)

    out_dir = Path(args.output_dir) / f"walk_forward_{datetime.now():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.assign(date=preds["date"].dt.strftime("%Y-%m-%d")).to_csv(out_dir / "predicoes_walk_forward.csv", index=False)
    summary.to_csv(out_dir / "metricas_por_periodo.csv", index=False)
    bootstrap.to_csv(out_dir / "bootstrap_knn_vs_rf.csv", index=False)
    extremes.to_csv(out_dir / "knn_probabilidades_extremas.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n===== Metricas por periodo =====")
        print(summary.to_string(index=False))
        print("\n===== Bootstrap em blocos (KNN - RF), walk-forward completo =====")
        print(bootstrap.to_string(index=False))
        print("\n===== KNN: probabilidades extremas =====")
        print(extremes.to_string(index=False))
    print(f"\nArquivos salvos em {out_dir}")


if __name__ == "__main__":
    main()
