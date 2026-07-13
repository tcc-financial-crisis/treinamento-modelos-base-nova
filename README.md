# Treinamento ML - Previsão de Crises Financeiras

## Como rodar

```bash
pip install -r requirements.txt
# Treino local (sem S3)
python treinamento.py --no-upload
# Treino + upload S3
python treinamento.py
# Apenas upload dos artefatos já gerados
python treinamento.py --upload-only --run-id YYYY-MM-DD
```

