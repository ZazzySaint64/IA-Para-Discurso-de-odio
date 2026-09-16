import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CATEGORIAS_TOXICIDADE = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
FEEDBACK_CSV = os.path.join(BASE, "feedback_treino.csv")


def carregar_dados(incluir_feedback=True):
    dados = pd.read_csv(os.path.join(BASE, "HateBR-7.0.0", "dataset", "HateBR.csv"))
    dados = dados[["comentario", "label_final"]]

    told_br = pd.read_csv(os.path.join(BASE, "HateBR-7.0.0", "dataset", "ToLD-BR.csv"))
    told_br["label_final"] = (told_br[CATEGORIAS_TOXICIDADE].sum(axis=1) >= 2).astype(int)
    told_br = told_br.rename(columns={"text": "comentario"})
    told_br = told_br[["comentario", "label_final"]]

    dados = pd.concat([dados, told_br], ignore_index=True)

    if incluir_feedback and os.path.exists(FEEDBACK_CSV):
        feedback = pd.read_csv(FEEDBACK_CSV)[["comentario", "label_final"]]
        dados = pd.concat([dados, feedback], ignore_index=True)

    return dados
