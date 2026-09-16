import hashlib
import hmac
import os
import time
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from dados_treino import BASE, FEEDBACK_CSV, carregar_dados

SENHA_HASH_ENV = "HATEBR_SENHA_HASH"
MAX_TENTATIVAS = 5
MODELO_PATH = os.path.join(BASE, "modelo_logistic_regression.pkl")
VETORIZADOR_PATH = os.path.join(BASE, "vetorizador.pkl")

st.title("Painel de Treinamento — HateBR")
st.caption("Área restrita. Aqui você ensina o modelo mostrando exemplos e dizendo se são ou não discurso de ódio.")

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False
if "tentativas" not in st.session_state:
    st.session_state.tentativas = 0

if not st.session_state.autenticado:
    hash_esperado = os.environ.get(SENHA_HASH_ENV)
    if not hash_esperado:
        st.error(
            f"Variável de ambiente {SENHA_HASH_ENV} não definida. "
            "Rode `python gerar_hash_senha.py` pra criar sua senha e ver como definir essa variável."
        )
        st.stop()

    if st.session_state.tentativas >= MAX_TENTATIVAS:
        st.error("Muitas tentativas erradas. Feche e abra esta página de novo pra tentar outra vez.")
        st.stop()

    senha_digitada = st.text_input(
        "Senha de acesso",
        type="password",
        help="Só quem sabe essa senha consegue ensinar o modelo.",
    )
    if st.button("Entrar", help="Confere a senha digitada e libera o painel de treino."):
        hash_digitado = hashlib.sha256(senha_digitada.encode("utf-8")).hexdigest()
        if hmac.compare_digest(hash_digitado, hash_esperado):
            st.session_state.autenticado = True
            st.session_state.tentativas = 0
            st.rerun()
        else:
            st.session_state.tentativas += 1
            time.sleep(1.5)
            st.error(f"Senha incorreta. Tentativa {st.session_state.tentativas} de {MAX_TENTATIVAS}.")
    st.stop()

col_status, col_sair = st.columns([4, 1])
col_status.success("Acesso liberado. Você já pode ensinar o modelo.")
if col_sair.button("Sair"):
    st.session_state.autenticado = False
    st.rerun()

modelo = joblib.load(MODELO_PATH)
vetorizador = joblib.load(VETORIZADOR_PATH)

st.header("1. Ensine um exemplo novo")
st.caption("Escreva um comentário e diga qual é o rótulo certo. Isso fica guardado até você mandar atualizar o modelo.")

comentario = st.text_area(
    "Comentário para analisar",
    help="Cole ou digite a frase que você quer ensinar ao modelo.",
)

if comentario.strip():
    predicao = modelo.predict(vetorizador.transform([comentario]))[0]
    palpite = "Discurso de Ódio" if predicao == 1 else "Não é Discurso de Ódio"
    st.write(f"Palpite do modelo atual: **{palpite}** (isso ainda não é o rótulo salvo, só uma prévia)")

rotulo = st.radio(
    "Qual é o rótulo certo para esse comentário?",
    ["Não é discurso de ódio", "É discurso de ódio"],
    help="Sua escolha aqui é o que o modelo vai aprender, mesmo que seja diferente do palpite acima.",
)

if st.button("Guardar este exemplo", help="Adiciona o comentário e o rótulo escolhido à lista de exemplos aguardando treino."):
    if not comentario.strip():
        st.warning("Digite um comentário antes de guardar.")
    else:
        label_final = 1 if rotulo == "É discurso de ódio" else 0
        linha = pd.DataFrame([{
            "comentario": comentario,
            "label_final": label_final,
            "data": datetime.now().isoformat(timespec="seconds"),
        }])
        cabecalho = not os.path.exists(FEEDBACK_CSV)
        linha.to_csv(FEEDBACK_CSV, mode="a", header=cabecalho, index=False)
        st.success("Exemplo guardado. Ele só entra no modelo quando você atualizar (passo 2).")

st.divider()
st.header("2. Atualize o modelo com os exemplos guardados")
st.caption(
    "Isso retreina o modelo do zero, usando o dataset original mais todos os exemplos que você guardou, "
    "e substitui o modelo atual. Pode demorar alguns segundos."
)

if os.path.exists(FEEDBACK_CSV):
    n_exemplos = len(pd.read_csv(FEEDBACK_CSV))
    st.write(f"Você guardou **{n_exemplos}** exemplo(s) que ainda não entraram no modelo.")
else:
    st.write("Nenhum exemplo guardado ainda. Ensine algo no passo 1 primeiro.")

if st.button("Atualizar modelo agora", help="Retreina e substitui o modelo atual usando os exemplos guardados."):
    with st.spinner("Atualizando o modelo... isso pode levar alguns segundos."):
        dados = carregar_dados()
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(class_weight="balanced")),
        ])
        validador = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, dados["comentario"], dados["label_final"], cv=validador, scoring="f1_macro"
        )
        pipeline.fit(dados["comentario"], dados["label_final"])
        joblib.dump(pipeline.named_steps["clf"], MODELO_PATH)
        joblib.dump(pipeline.named_steps["tfidf"], VETORIZADOR_PATH)
    st.success(f"Modelo atualizado com {len(dados)} exemplos no total. Qualidade estimada (F1 macro): {scores.mean():.4f}")
    st.rerun()

# python -m streamlit run painel_treino.py (CÓDIGO DE EXECUÇÃO)
