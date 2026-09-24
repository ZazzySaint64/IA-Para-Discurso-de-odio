"""Mede o pipeline treinado contra as frases curtas de avaliação.

ml/avaliacao_frases_curtas.csv nunca entra em `carregar_dados` nem em
`ml/frases_curtas.csv` (treino): é a medida honesta, feita em frases que o
modelo nunca viu.
"""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

CAMINHO_AVALIACAO_PADRAO = Path("ml/avaliacao_frases_curtas.csv")


def avaliar_frases_curtas(pipeline: Pipeline, caminho: Path = CAMINHO_AVALIACAO_PADRAO) -> float:
    """Acurácia do pipeline no CSV de avaliação. Devolve NaN se o arquivo não existir."""
    caminho = Path(caminho)
    if not caminho.exists():
        return float("nan")
    df = pd.read_csv(caminho)
    previstos = pipeline.predict(df["comentario"])
    return float((previstos == df["label_final"]).mean())
