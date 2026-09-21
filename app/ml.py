"""Carrega o pipeline treinado e responde predições.

Este módulo NÃO treina nada. Treino é responsabilidade de `ml/treinar.py`,
que roda fora do processo da API.
"""

from pathlib import Path

import joblib

_pipeline = None


class ModeloIndisponivelError(RuntimeError):
    """Predição pedida antes de o modelo ser carregado."""


def carregar_modelo(caminho: Path) -> None:
    global _pipeline
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}")
    _pipeline = joblib.load(caminho)


def modelo_carregado() -> bool:
    return _pipeline is not None


def prever(texto: str) -> tuple[int, float]:
    """Classifica um texto. Devolve (label, confianca)."""
    if _pipeline is None:
        raise ModeloIndisponivelError("Modelo não carregado")
    label = int(_pipeline.predict([texto])[0])
    confianca = float(_pipeline.predict_proba([texto])[0].max())
    return label, confianca
