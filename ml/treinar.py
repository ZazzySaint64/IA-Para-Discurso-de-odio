"""Treina o classificador e substitui o artefato apenas se o F1 melhorar.

Sucessor do retreinar_modelo.py. Roda FORA do processo da API: precisa de
muita RAM por alguns minutos, o oposto do perfil de uma API.

Uso: python -m ml.treinar
"""

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy.orm import Session

from ml.avaliar import avaliar_frases_curtas
from ml.dados import PASTA_PADRAO, carregar_dados, carregar_frases_curtas

CAMINHO_ARTEFATO = Path("ml/artefatos/modelo.pkl")

# Medido em ml/frases_curtas.csv x ml/avaliacao_frases_curtas.csv (pesos 1, 5,
# 20): 20 é o que dá a melhor acurácia nas frases curtas sem derrubar o F1 no
# holdout dos dados originais. Ver relatório em
# .superpowers/sdd/2026-09-21-hatebr-api/modelo-frases-curtas-report.md
PESO_CURADAS_PADRAO = 20.0


def construir_pipeline() -> Pipeline:
    """TF-IDF de palavra + TF-IDF de char n-gram.

    Sozinho, o TF-IDF de palavra com vocabulário limitado descarta bigramas
    raros como "te odeio" ou palavras como "morra" (poucas ocorrências no
    dataset). O char_wb pega esses padrões pelos n-gramas de caractere mesmo
    quando a palavra/bigrama exata não sobra no vocabulário.
    """
    vetorizador = FeatureUnion(
        [
            ("palavra", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    sublinear_tf=True,
                    max_features=150000,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", vetorizador),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )


def treinar(
    db: Session | None = None,
    pasta_datasets: Path = PASTA_PADRAO,
    caminho_artefato: Path = CAMINHO_ARTEFATO,
    f1_atual: float | None = None,
    peso_curadas: float = PESO_CURADAS_PADRAO,
) -> dict:
    dados = carregar_dados(db=db, pasta_datasets=pasta_datasets)
    curadas = carregar_frases_curtas()
    seed = random.randint(0, 999_999)
    pipeline = construir_pipeline()

    # CV só nos dados originais (HateBR + ToLD-BR, mais exemplos ensinados
    # quando há banco) — a mesma composição usada pra medir a tabela de
    # referência do pipeline. As frases curadas são poucas e fáceis demais
    # (curtas, vocabulário bem separado); somá-las às dobras infla o F1 sem
    # dizer nada sobre generalização, então ficam de fora do CV e só entram
    # no fit final, com peso maior.
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
    acuracia_frases_curtas = None
    if substituiu:
        dados_completo = pd.concat([dados, curadas], ignore_index=True)
        pesos = np.concatenate([np.ones(len(dados)), np.full(len(curadas), peso_curadas)])
        pipeline.fit(
            dados_completo["comentario"],
            dados_completo["label_final"],
            clf__sample_weight=pesos,
        )
        acuracia_frases_curtas = avaliar_frases_curtas(pipeline)
        caminho_artefato.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, caminho_artefato)

    return {
        "f1_macro": f1_macro,
        "desvio": desvio,
        "seed": seed,
        "qtd_exemplos": len(dados) + len(curadas),
        "substituiu": substituiu,
        "acuracia_frases_curtas": acuracia_frases_curtas,
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
    if resultado["acuracia_frases_curtas"] is not None:
        print(f"Acurácia frases curtas (held-out): {resultado['acuracia_frases_curtas']:.4f}")
    print("Artefato substituído." if resultado["substituiu"] else "Artefato mantido.")


if __name__ == "__main__":
    main()
