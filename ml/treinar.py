"""Treina o classificador e substitui o artefato apenas se o F1 melhorar.

Sucessor do retreinar_modelo.py. Roda FORA do processo da API: precisa de
muita RAM por alguns minutos, o oposto do perfil de uma API.

Uso: python -m ml.treinar
"""

import random
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from ml.dados import PASTA_PADRAO, carregar_dados

CAMINHO_ARTEFATO = Path("ml/artefatos/modelo.pkl")


def construir_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def treinar(
    db: Session | None = None,
    pasta_datasets: Path = PASTA_PADRAO,
    caminho_artefato: Path = CAMINHO_ARTEFATO,
    f1_atual: float | None = None,
) -> dict:
    dados = carregar_dados(db=db, pasta_datasets=pasta_datasets)
    seed = random.randint(0, 999_999)
    pipeline = construir_pipeline()

    validador = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        dados["comentario"],
        dados["label_final"],
        cv=validador,
        scoring="f1_macro",
    )
    f1_macro = float(scores.mean())
    desvio = float(scores.std())

    substituiu = f1_atual is None or f1_macro > f1_atual
    if substituiu:
        pipeline.fit(dados["comentario"], dados["label_final"])
        caminho_artefato.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, caminho_artefato)

    return {
        "f1_macro": f1_macro,
        "desvio": desvio,
        "seed": seed,
        "qtd_exemplos": len(dados),
        "substituiu": substituiu,
    }


def main() -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        resultado = treinar(db=db)
    finally:
        db.close()
    print(f"F1 macro: {resultado['f1_macro']:.4f} (desvio: {resultado['desvio']:.4f})")
    print(f"Exemplos: {resultado['qtd_exemplos']}")
    print("Artefato substituído." if resultado["substituiu"] else "Artefato mantido.")


if __name__ == "__main__":
    main()
