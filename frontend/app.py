"""Cliente Streamlit da API de detecção de discurso de ódio.

Este arquivo NÃO carrega o modelo e NÃO acessa o banco. Ele só fala HTTP
com a API. Foi exatamente essa separação que motivou o redesenho do projeto.
"""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60  # o plano gratuito do Render hiberna; a primeira chamada demora

st.set_page_config(page_title="Detector de Discurso de Ódio", page_icon="🛡️")


def pedir(metodo: str, rota: str, **kwargs) -> requests.Response | None:
    try:
        return requests.request(metodo, f"{API}{rota}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Não consegui falar com a API: {exc}")
        return None


aba_publica, aba_treino = st.tabs(["Classificar", "Painel de treino"])

with aba_publica:
    st.title("Detector de Discurso de Ódio")
    texto = st.text_area("Comentário", max_chars=1000)
    if st.button("Classificar", disabled=not texto.strip()):
        resposta = pedir("POST", "/predicoes", json={"texto": texto})
        if resposta is None:
            pass
        elif resposta.status_code == 201:
            dado = resposta.json()
            if dado["label"] == 1:
                st.error(f"{dado['rotulo']} — confiança {dado['confianca']:.0%}")
            else:
                st.success(f"{dado['rotulo']} — confiança {dado['confianca']:.0%}")
        else:
            st.warning(resposta.json().get("detail", "Erro inesperado"))

    metricas = pedir("GET", "/metricas")
    if metricas is not None and metricas.status_code == 200:
        m = metricas.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Classificações feitas", m["total_predicoes"])
        c2.metric("Taxa de ódio", f"{m['taxa_odio']:.0%}")
        c3.metric("F1 do modelo", f"{m['f1_modelo']:.3f}" if m["f1_modelo"] else "—")

with aba_treino:
    st.title("Painel de treino")

    if "token" not in st.session_state:
        st.session_state.token = None

    if st.session_state.token is None:
        with st.form("login"):
            email = st.text_input("Email")
            senha = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                resposta = pedir("POST", "/auth/login", data={"username": email, "password": senha})
                if resposta is not None and resposta.status_code == 200:
                    st.session_state.token = resposta.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Email ou senha incorretos.")
    else:
        cabecalho = {"Authorization": f"Bearer {st.session_state.token}"}

        with st.form("ensinar"):
            novo = st.text_area("Comentário para ensinar", max_chars=1000)
            rotulo = st.radio("É discurso de ódio?", ["Não", "Sim"], horizontal=True)
            if st.form_submit_button("Guardar este exemplo"):
                resposta = pedir(
                    "POST",
                    "/exemplos",
                    json={"texto": novo, "label": 1 if rotulo == "Sim" else 0},
                    headers=cabecalho,
                )
                if resposta is None:
                    pass
                elif resposta.status_code == 201:
                    st.success("Exemplo guardado.")
                elif resposta.status_code == 409:
                    st.warning("Esse texto já foi ensinado antes.")
                elif resposta.status_code == 401:
                    st.session_state.token = None
                    st.warning("Sessão expirada, entre de novo.")
                    st.rerun()
                else:
                    st.error(resposta.json().get("detail", "Erro inesperado"))

        if st.button("Atualizar modelo agora"):
            resposta = pedir("POST", "/treinos", headers=cabecalho)
            if resposta is None:
                pass
            elif resposta.status_code == 202:
                st.info(f"Treino iniciado (id {resposta.json()['id']}).")
            else:
                st.warning(resposta.json().get("detail", "Erro inesperado"))

        exemplos = pedir("GET", "/exemplos?limite=20", headers=cabecalho)
        if exemplos is not None and exemplos.status_code == 200:
            st.subheader(f"Exemplos ensinados ({exemplos.json()['total']})")
            st.dataframe(exemplos.json()["itens"], use_container_width=True)

        if st.button("Sair"):
            st.session_state.token = None
            st.rerun()
