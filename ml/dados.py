"""Monta o conjunto de treino: HateBR + ToLD-BR + exemplos ensinados no painel.

Sucessor do dados_treino.py. A diferença é a origem dos exemplos próprios:
antes um CSV, agora a tabela `exemplo`.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

CATEGORIAS_TOXICIDADE = [
    "homophobia",
    "obscene",
    "insult",
    "racism",
    "misogyny",
    "xenophobia",
]
PASTA_PADRAO = Path("HateBR-7.0.0/dataset")
CAMINHO_CURADAS_PADRAO = Path("ml/frases_curtas.csv")


def carregar_dados(db: Session | None = None, pasta_datasets: Path = PASTA_PADRAO) -> pd.DataFrame:
    pasta_datasets = Path(pasta_datasets)

    caminho_hatebr = pasta_datasets / "HateBR.csv"
    if not caminho_hatebr.exists():
        raise FileNotFoundError(
            f"HateBR.csv não encontrado em {pasta_datasets}. "
            "Baixe em https://github.com/franciellevargas/HateBR"
        )
    hatebr = pd.read_csv(caminho_hatebr)[["comentario", "label_final"]]

    caminho_told = pasta_datasets / "ToLD-BR.csv"
    if not caminho_told.exists():
        raise FileNotFoundError(
            f"ToLD-BR.csv não encontrado em {pasta_datasets}. "
            "Baixe em https://github.com/JAugusto97/ToLD-Br"
        )
    told = pd.read_csv(caminho_told)
    told["label_final"] = (told[CATEGORIAS_TOXICIDADE].sum(axis=1) >= 2).astype(int)
    told = told.rename(columns={"text": "comentario"})[["comentario", "label_final"]]

    partes = [hatebr, told]

    if db is not None:
        from app.models import Exemplo

        linhas = db.execute(select(Exemplo.texto, Exemplo.label)).all()
        if linhas:
            partes.append(pd.DataFrame(linhas, columns=["comentario", "label_final"]))

    return pd.concat(partes, ignore_index=True)


def carregar_frases_curtas(caminho: Path = CAMINHO_CURADAS_PADRAO) -> pd.DataFrame:
    """Frases curtas curadas à mão (ml/frases_curtas.csv).

    HateBR e ToLD-BR são comentários longos de Instagram político: hostilidade
    curta dirigida a uma pessoa ("eu te odeio", "morra") quase não aparece
    neles, então o modelo nunca aprende esse padrão. Este CSV existe só pra
    cobrir esse buraco. Separado de `carregar_dados` de propósito: ele entra
    no treino com peso maior (ver `ml/treinar.py`) e NUNCA no CV — ver
    `treinar()` para o motivo.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return pd.DataFrame(columns=["comentario", "label_final"])
    return pd.read_csv(caminho)[["comentario", "label_final"]]
