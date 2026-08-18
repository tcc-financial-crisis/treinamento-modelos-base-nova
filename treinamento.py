#!/usr/bin/env python3
"""
Pipeline de treinamento de modelos para predicao de crise financeira (target_12m).

Encapsula o fluxo do notebook treinamento_todas_features.ipynb:
- 14 features (series base + variacoes diff/mom/yoy)
- split temporal antes do preprocessamento (evita vazamento)
- StandardScaler e PCA ajustados apenas no conjunto de treino
- GridSearchCV paralelo (TimeSeriesSplit)
- artefatos persistidos em disco (joblib) e enviados ao S3
- pickle do proprio script gerado ao final
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import joblib
import matplotlib
from botocore.exceptions import ClientError, NoCredentialsError

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
PCA_FEATURES = ["pca_1", "pca_2", "pca_3"]
LABELS = ["Normal", "Crise"]
ALL_CLASSES = [0, 1]
DEFAULT_BUCKET = "client-789763302807-us-east-1-an"
DEFAULT_DATASET = "./raw/dataset_merged_binary.csv"
SCRIPT_PATH = Path(__file__).resolve()


@dataclass
class TrainingConfig:
    dataset_path: Path = Path(DEFAULT_DATASET)
    output_dir: Path = Path("./output")
    bucket: str = DEFAULT_BUCKET
    s3_prefix: str = "models"
    test_size: float = 0.3
    random_state: int = 42
    n_splits: int = 5
    upload_s3: bool = True
    save_plots: bool = True
    pickle_script: bool = True


@dataclass
class TrainingContext:
    config: TrainingConfig
    run_id: str
    run_dir: Path
    resultados: list[dict[str, Any]] = field(default_factory=list)
    predicoes: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    best_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def _br_date() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%d")


def _memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _timed_fit(estimator, X_train, y_train) -> tuple[Any, float]:
    start = time.time()
    estimator.fit(X_train, y_train)
    return estimator, time.time() - start


def _timed_predict(model, X_test) -> tuple[np.ndarray, float]:
    start = time.time()
    preds = model.predict(X_test)
    return preds, time.time() - start


def load_and_split(config: TrainingConfig) -> tuple[dict[str, Any], StandardScaler, PCA]:
    cols = ["date", *FEATURES, "target_12m"]
    df = pd.read_csv(config.dataset_path, parse_dates=["date"])
    df = df[cols].dropna().sort_values("date").reset_index(drop=True)

    X = df[FEATURES]
    y = df["target_12m"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=False,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=3)
    X_train_pca_arr = pca.fit_transform(X_train_scaled)
    X_test_pca_arr = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_
    print(
        f"PCA (ajustado no treino): PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, "
        f"PC3={explained[2]:.2%}, total={explained.sum():.2%}"
    )

    X_train_pca = pd.DataFrame(
        X_train_pca_arr,
        columns=PCA_FEATURES,
        index=X_train_raw.index,
    )
    X_test_pca = pd.DataFrame(
        X_test_pca_arr,
        columns=PCA_FEATURES,
        index=X_test_raw.index,
    )

    dates_train = df.loc[X_train_raw.index, "date"].reset_index(drop=True)
    dates_test = df.loc[X_test_raw.index, "date"].reset_index(drop=True)

    split_base = {
        "dates_train": dates_train,
        "dates_test": dates_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }

    splits = {
        "no_pca": {
            **split_base,
            "X_train": X_train_raw.reset_index(drop=True),
            "X_test": X_test_raw.reset_index(drop=True),
        },
        "pca": {
            **split_base,
            "X_train": X_train_pca.reset_index(drop=True),
            "X_test": X_test_pca.reset_index(drop=True),
        },
    }
    return splits, scaler, pca


def _save_confusion_matrix(
    ctx: TrainingContext,
    y_true,
    y_pred,
    title: str,
    filename: str,
) -> None:
    if not ctx.config.save_plots:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=45)
    ax.set_title(title)
    path = ctx.run_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    ctx.artifacts.append(path)


def _label_name(value: int) -> str:
    return LABELS[int(value)]


def _positive_proba(model, X) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)[:, 1]


def _append_predictions(
    ctx: TrainingContext,
    nome_modelo: str,
    conjunto: str,
    dates: pd.Series,
    y_true,
    y_pred,
    prob_crise: np.ndarray | None = None,
) -> None:
    for i, (date, yt, yp) in enumerate(zip(dates, y_true, y_pred)):
        registro = {
            "modelo": nome_modelo,
            "conjunto": conjunto,
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_true_label": _label_name(yt),
            "y_pred_label": _label_name(yp),
            "acerto": int(yt == yp),
        }
        if prob_crise is not None:
            registro["prob_crise"] = round(float(prob_crise[i]), 6)
        ctx.predicoes.append(registro)


def _register_result(
    ctx: TrainingContext,
    nome_modelo: str,
    y_test,
    y_pred_test,
    tempo_treino: float,
    tempo_predicao: float,
    y_train=None,
    y_pred_train=None,
    dates_train: pd.Series | None = None,
    dates_test: pd.Series | None = None,
    prob_train: np.ndarray | None = None,
    prob_test: np.ndarray | None = None,
) -> None:
    ctx.resultados.append(
        {
            "modelo": nome_modelo,
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(
                y_test, y_pred_test, average="macro", labels=ALL_CLASSES, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred_test, average="macro", labels=ALL_CLASSES, zero_division=0
            ),
            "f1_macro": f1_score(
                y_test, y_pred_test, average="macro", labels=ALL_CLASSES, zero_division=0
            ),
            "tempo_treino_s": round(tempo_treino, 4),
            "tempo_predicao_ms": round(tempo_predicao * 1000, 4),
            "memoria_mb": round(_memory_mb(), 2),
        }
    )

    print(f"\n===== {nome_modelo} =====")
    if y_train is not None and y_pred_train is not None:
        print("\n--- TREINO ---")
        print(
            classification_report(
                y_train, y_pred_train, labels=ALL_CLASSES, target_names=LABELS, zero_division=0
            )
        )
    print("\n--- TESTE ---")
    print(
        classification_report(
            y_test, y_pred_test, labels=ALL_CLASSES, target_names=LABELS, zero_division=0
        )
    )
    print(f"Tempo treino: {tempo_treino:.4f}s | Predicao: {tempo_predicao * 1000:.4f}ms")

    if dates_train is not None and y_train is not None and y_pred_train is not None:
        _append_predictions(
            ctx, nome_modelo, "treino", dates_train, y_train, y_pred_train, prob_train
        )
    if dates_test is not None:
        _append_predictions(ctx, nome_modelo, "teste", dates_test, y_test, y_pred_test, prob_test)

    slug = nome_modelo.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _save_confusion_matrix(ctx, y_test, y_pred_test, f"{nome_modelo} - Teste", f"cm_test_{slug}.png")
    if y_train is not None and y_pred_train is not None:
        _save_confusion_matrix(
            ctx, y_train, y_pred_train, f"{nome_modelo} - Treino", f"cm_train_{slug}.png"
        )


def _persist_model(ctx: TrainingContext, name: str, model) -> Path:
    path = ctx.run_dir / f"{name}.joblib"
    joblib.dump(model, path)
    ctx.artifacts.append(path)
    return path


def _grid_search(
    ctx: TrainingContext,
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    tscv: TimeSeriesSplit,
    label: str,
) -> Any:
    grid = GridSearchCV(estimator, param_grid, cv=tscv, scoring="f1_macro", n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train)
    tempo = time.time() - start
    ctx.best_params[label] = grid.best_params_
    print(f"Melhores parametros ({label}): {grid.best_params_}")
    return grid.best_estimator_, tempo


def train_decision_tree(ctx: TrainingContext, splits: dict, tscv: TimeSeriesSplit) -> None:
    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "ccp_alpha": [0.0, 0.001, 0.005],
    }
    base = DecisionTreeClassifier(random_state=ctx.config.random_state)

    for variant, label in (("no_pca", "sem PCA"), ("pca", "PCA")):
        data = splits[variant]
        model, tempo = _grid_search(
            ctx, base, param_grid, data["X_train"], data["y_train"], tscv, f"decision_tree_{variant}"
        )
        y_pred_train = model.predict(data["X_train"])
        y_pred_test, tempo_pred = _timed_predict(model, data["X_test"])
        _register_result(
            ctx,
            f"Decision Tree ({label})",
            data["y_test"],
            y_pred_test,
            tempo,
            tempo_pred,
            data["y_train"],
            y_pred_train,
            data["dates_train"],
            data["dates_test"],
            _positive_proba(model, data["X_train"]),
            _positive_proba(model, data["X_test"]),
        )
        _persist_model(ctx, f"decision_tree_{variant}", model)


def train_knn(ctx: TrainingContext, splits: dict, tscv: TimeSeriesSplit) -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )
    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    for variant, label in (("no_pca", "sem PCA"), ("pca", "PCA")):
        data = splits[variant]
        model, tempo = _grid_search(
            ctx, pipeline, param_grid, data["X_train"], data["y_train"], tscv, f"knn_{variant}"
        )
        y_pred_train = model.predict(data["X_train"])
        y_pred_test, tempo_pred = _timed_predict(model, data["X_test"])
        _register_result(
            ctx,
            f"KNN ({label})",
            data["y_test"],
            y_pred_test,
            tempo,
            tempo_pred,
            data["y_train"],
            y_pred_train,
            data["dates_train"],
            data["dates_test"],
            _positive_proba(model, data["X_train"]),
            _positive_proba(model, data["X_test"]),
        )
        _persist_model(ctx, f"knn_{variant}", model)


def train_voting(ctx: TrainingContext, splits: dict) -> None:
    knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))])
    tree = DecisionTreeClassifier(random_state=ctx.config.random_state, max_depth=5)
    rf = RandomForestClassifier(n_estimators=200, random_state=ctx.config.random_state)
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    estimators = [("knn", knn), ("tree", tree), ("rf", rf), ("logreg", logreg)]

    for variant, label in (("no_pca", "sem PCA"), ("pca", "PCA")):
        data = splits[variant]
        model = VotingClassifier(estimators=estimators, voting="soft")
        model, tempo = _timed_fit(model, data["X_train"], data["y_train"])
        y_pred_train = model.predict(data["X_train"])
        y_pred_test, tempo_pred = _timed_predict(model, data["X_test"])
        _register_result(
            ctx,
            f"Voting Ensemble ({label})",
            data["y_test"],
            y_pred_test,
            tempo,
            tempo_pred,
            data["y_train"],
            y_pred_train,
            data["dates_train"],
            data["dates_test"],
            _positive_proba(model, data["X_train"]),
            _positive_proba(model, data["X_test"]),
        )
        _persist_model(ctx, f"voting_{variant}", model)


def _save_feature_importance(
    ctx: TrainingContext,
    features: list[str],
    importances: np.ndarray,
    title: str,
    filename: str,
) -> None:
    df_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    csv_path = ctx.run_dir / f"{filename}.csv"
    df_imp.to_csv(csv_path, index=False)
    ctx.artifacts.append(csv_path)
    print(df_imp.to_string(index=False))

    if not ctx.config.save_plots:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    png_path = ctx.run_dir / f"{filename}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    ctx.artifacts.append(png_path)


def train_random_forest(ctx: TrainingContext, splits: dict, tscv: TimeSeriesSplit) -> None:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }
    base = RandomForestClassifier(random_state=ctx.config.random_state)

    for variant, label, feat_names in (
        ("no_pca", "sem PCA", FEATURES),
        ("pca", "PCA", PCA_FEATURES),
    ):
        data = splits[variant]
        model, tempo = _grid_search(
            ctx, base, param_grid, data["X_train"], data["y_train"], tscv, f"random_forest_{variant}"
        )
        y_pred_train = model.predict(data["X_train"])
        y_pred_test, tempo_pred = _timed_predict(model, data["X_test"])
        _register_result(
            ctx,
            f"Random Forest ({label})",
            data["y_test"],
            y_pred_test,
            tempo,
            tempo_pred,
            data["y_train"],
            y_pred_train,
            data["dates_train"],
            data["dates_test"],
            _positive_proba(model, data["X_train"]),
            _positive_proba(model, data["X_test"]),
        )
        _persist_model(ctx, f"random_forest_{variant}", model)
        print(f"\nImportancia das variaveis ({label}):")
        _save_feature_importance(
            ctx,
            feat_names,
            model.feature_importances_,
            f"Importancia - Random Forest ({label})",
            f"rf_importance_{variant}",
        )


def save_predictions(ctx: TrainingContext) -> Path:
    df_predicoes = pd.DataFrame(ctx.predicoes)
    csv_path = ctx.run_dir / "predicoes.csv"
    json_path = ctx.run_dir / "predicoes.json"
    teste_path = ctx.run_dir / "predicoes_teste.csv"

    df_predicoes.to_csv(csv_path, index=False)
    df_predicoes.to_json(json_path, orient="records", indent=2, force_ascii=False)

    df_teste = df_predicoes[df_predicoes["conjunto"] == "teste"].copy()
    df_teste.to_csv(teste_path, index=False)

    ctx.artifacts.extend([csv_path, json_path, teste_path])
    print(f"\nPrevisoes salvas: {csv_path} ({len(df_predicoes)} registros)")
    return csv_path


def save_summary(ctx: TrainingContext) -> Path:
    df_resultados = pd.DataFrame(ctx.resultados).round(
        {"accuracy": 4, "precision": 4, "recall": 4, "f1_macro": 4, "tempo_treino_s": 4}
    )
    csv_path = ctx.run_dir / "resultados.csv"
    json_path = ctx.run_dir / "resultados.json"
    meta_path = ctx.run_dir / "metadata.json"

    df_resultados.to_csv(csv_path, index=False)
    df_resultados.to_json(json_path, orient="records", indent=2)

    metadata = {
        "run_id": ctx.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(ctx.config.dataset_path),
        "features": FEATURES,
        "target": "target_12m",
        "preprocessing": "StandardScaler e PCA ajustados apenas no treino",
        "best_params": ctx.best_params,
        "resultados": ctx.resultados,
        "predicoes_count": len(ctx.predicoes),
        "predicoes_files": ["predicoes.csv", "predicoes.json", "predicoes_teste.csv"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    ctx.artifacts.extend([csv_path, json_path, meta_path])
    print("\n===== RESULTADOS FINAIS =====")
    print(df_resultados.to_string(index=False))
    return csv_path


def save_comparison_plot(ctx: TrainingContext) -> None:
    if not ctx.config.save_plots:
        return

    df_resultados = pd.DataFrame(ctx.resultados).set_index("modelo")
    metricas = ["accuracy", "precision", "recall", "f1_macro"]
    width = 0.2

    for group_name, filtro in (
        ("sem_pca", lambda idx: "sem PCA" in idx),
        ("com_pca", lambda idx: "sem PCA" not in idx),
    ):
        subset = df_resultados[df_resultados.index.map(filtro)]
        if subset.empty:
            continue

        x = np.arange(len(subset))
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, metrica in enumerate(metricas):
            ax.bar(x + i * width, subset[metrica], width, label=metrica)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(subset.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Comparacao de Modelos - {'Sem PCA' if group_name == 'sem_pca' else 'Com PCA'}")
        ax.legend()
        fig.tight_layout()
        path = ctx.run_dir / f"comparacao_{group_name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        ctx.artifacts.append(path)


def gerar_pickle_script(ctx: TrainingContext) -> Path:
    payload = {
        "filename": SCRIPT_PATH.name,
        "source": SCRIPT_PATH.read_text(encoding="utf-8"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "entrypoint": "treinamento.main",
    }
    path = ctx.run_dir / "treinamento_script.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    ctx.artifacts.append(path)
    print(f"Pickle do script gerado: {path}")
    return path


def upload_to_s3(ctx: TrainingContext) -> list[str]:
    if not ctx.config.upload_s3:
        return []

    s3 = boto3.client("s3")
    prefix = f"{ctx.config.s3_prefix}/{ctx.run_id}/"
    uploaded: list[str] = []

    try:
        for path in sorted(set(ctx.artifacts)):
            if not path.exists():
                continue
            key = f"{prefix}{path.name}"
            s3.upload_file(str(path), ctx.config.bucket, key)
            uploaded.append(f"s3://{ctx.config.bucket}/{key}")
            print(f"Enviado: s3://{ctx.config.bucket}/{key}")

        manifest = {
            "run_id": ctx.run_id,
            "bucket": ctx.config.bucket,
            "prefix": prefix,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "objects": uploaded,
        }
        manifest_key = f"{prefix}manifest.json"
        s3.put_object(
            Bucket=ctx.config.bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uploaded.append(f"s3://{ctx.config.bucket}/{manifest_key}")
    except NoCredentialsError:
        print(
            "ERRO: credenciais AWS nao encontradas. Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "ou execute `aws configure`. Artefatos salvos localmente em "
            f"{ctx.run_dir}. Use --upload-only para enviar depois.",
            file=sys.stderr,
        )
    except ClientError as exc:
        print(f"ERRO no upload S3: {exc}", file=sys.stderr)

    return uploaded


def upload_existing_run(config: TrainingConfig, run_id: str) -> dict[str, Any]:
    run_dir = config.output_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Diretorio de execucao nao encontrado: {run_dir}")

    artifacts = [p for p in run_dir.iterdir() if p.is_file()]
    ctx = TrainingContext(config=config, run_id=run_id, run_dir=run_dir, artifacts=artifacts)
    uploaded = upload_to_s3(ctx)
    return {"run_id": run_id, "output_dir": str(run_dir), "uploaded": uploaded}


def run_training(config: TrainingConfig) -> dict[str, Any]:
    run_id = _br_date()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = TrainingContext(config=config, run_id=run_id, run_dir=run_dir)
    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    splits, scaler, pca = load_and_split(config)

    joblib.dump(scaler, run_dir / "scaler.joblib")
    joblib.dump(pca, run_dir / "pca.joblib")
    ctx.artifacts.extend([run_dir / "scaler.joblib", run_dir / "pca.joblib"])

    train_decision_tree(ctx, splits, tscv)
    train_knn(ctx, splits, tscv)
    train_voting(ctx, splits)
    train_random_forest(ctx, splits, tscv)

    save_predictions(ctx)
    save_summary(ctx)
    save_comparison_plot(ctx)

    if config.pickle_script:
        gerar_pickle_script(ctx)

    uploaded = upload_to_s3(ctx)
    return {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "artifacts": [str(p) for p in ctx.artifacts],
        "uploaded": uploaded,
    }


def parse_args() -> tuple[TrainingConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Treinamento de modelos de crise financeira")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--bucket", default=os.environ.get("CLIENT_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--s3-prefix", default=os.environ.get("MODELS_PREFIX", "models"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--run-id", default=_br_date())
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-pickle-script", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        upload_s3=not args.no_upload,
        save_plots=not args.no_plots,
        pickle_script=not args.no_pickle_script,
    )
    return config, args


def main() -> None:
    config, args = parse_args()

    if args.upload_only:
        summary = upload_existing_run(config, args.run_id)
    else:
        summary = run_training(config)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
