# Treinamento ML - Previsão de Crises Financeiras

Projeto de treinamento de modelos de classificação binária para prever crises financeiras com horizonte de 12 meses (`target_12m`), usando indicadores macroeconômicos e suas variáveis derivadas.

O script `treinamento.py` encapsula o fluxo do notebook `treinamento_todas_features.ipynb`.

## Objetivo

Classificar cada mês como **Normal** (0) ou **Crise** (1) com base em indicadores como spread de crédito, produção industrial, inflação, juros reais, VIX e inclinação da curva de juros — incluindo variações temporais (diff, mom, yoy).

## Features utilizadas (14)

| Grupo | Variáveis |
|-------|-----------|
| Crédito | `credit_spread`, `credit_spread_diff` |
| Produção | `industrial_production`, `industrial_production_mom`, `industrial_production_yoy` |
| Inflação | `price_index`, `inflation_mom`, `inflation_yoy` |
| Juros | `real_interest`, `real_interest_diff` |
| Volatilidade | `volatility_index`, `volatility_index_mom` |
| Curva de juros | `yield_curve_slope`, `yield_curve_slope_diff` |

## Modelos treinados

Para cada algoritmo, são treinadas duas variantes: **sem PCA** (features originais) e **com PCA** (3 componentes):

- Decision Tree (`GridSearchCV` + `ccp_alpha`)
- KNN (pipeline com `StandardScaler`)
- Voting Ensemble (KNN + Tree + Random Forest + Logistic Regression)
- Random Forest (`GridSearchCV`)

## Fluxo de preprocessamento

Diferente da versão anterior com 6 features, o pipeline segue a metodologia correta do notebook:

1. Split temporal 70/30 **sem shuffle**
2. `StandardScaler` ajustado **apenas no treino**
3. `PCA` (3 componentes) ajustado **apenas no treino escalonado**
4. Transformação do conjunto de teste com os objetos já ajustados

Isso evita vazamento de informação do teste para o treino.

## Estrutura do repositório

```
treinamento-modelos-base-nova/
├── raw/
│   └── dataset_merged_binary.csv   # Dataset de entrada
├── treinamento.py                  # Script principal
├── treinamento_todas_features.ipynb
├── treinamento.ipynb               # Versão original (6 features)
├── requirements.txt
└── output/                         # Artefatos gerados (gitignored)
    └── YYYY-MM-DD/
```

## Como rodar

```bash
pip install -r requirements.txt

# Treino local (sem S3)
python treinamento.py --no-upload

# Treino completo + upload S3
python treinamento.py

# Apenas upload de uma execução já gerada
python treinamento.py --upload-only --run-id YYYY-MM-DD
```

### Argumentos CLI

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--dataset` | Caminho do CSV | `./raw/dataset_merged_binary.csv` |
| `--output-dir` | Diretório de saída | `./output` |
| `--bucket` | Bucket S3 de destino | `client-789763302807-us-east-1-an` |
| `--s3-prefix` | Prefixo no bucket | `models` |
| `--no-upload` | Desabilita upload S3 | — |
| `--upload-only` | Envia artefatos existentes | — |
| `--run-id` | ID da execução (data BR) | data atual |
| `--no-plots` | Não gera gráficos PNG | — |
| `--no-pickle-script` | Não gera pickle do script | — |

## Artefatos gerados

Cada execução cria `output/YYYY-MM-DD/` com:

- **Pré-processamento**: `scaler.joblib`, `pca.joblib`
- **Modelos**: `decision_tree_*.joblib`, `knn_*.joblib`, `voting_*.joblib`, `random_forest_*.joblib`
- **Métricas**: `resultados.csv`, `resultados.json`, `metadata.json`
- **Previsões**: `predicoes.csv` (treino + teste), `predicoes_teste.csv` (somente teste), `predicoes.json`
- **Visualizações**: matrizes de confusão, comparação de modelos, importância de features
- **Deploy**: `treinamento_script.pkl` (código-fonte serializado)

## Upload S3

Os artefatos são enviados para:

```
s3://client-789763302807-us-east-1-an/models/YYYY-MM-DD/
```

Requer credenciais AWS configuradas (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` ou `aws configure`).

Variáveis de ambiente opcionais:

- `CLIENT_BUCKET` — sobrescreve o bucket padrão
- `MODELS_PREFIX` — sobrescreve o prefixo S3

## Dependências

- numpy, pandas, matplotlib, scikit-learn, psutil, boto3, joblib
