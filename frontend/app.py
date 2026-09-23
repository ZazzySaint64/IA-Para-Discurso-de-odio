"""Cliente Streamlit da API de detecção de discurso de ódio.

Este arquivo NÃO carrega o modelo e NÃO acessa o banco. Ele só fala HTTP
com a API. Foi exatamente essa separação que motivou o redesenho do projeto.
"""

import os
import time

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60  # teto por chamada; a espera pela hibernação é o retry em pedir(), não isto


def pedir(metodo: str, rota: str, **kwargs) -> requests.Response | None:
    """502/503/504 aqui não é "a API caiu": é o Render acordando o container
    hibernado, e o gateway responde esse erro na hora, sem esperar o container
    subir — por isso TIMEOUT nunca ajudava. Insiste por ~90s (sobra pra
    hibernação acabar) antes de devolver a última tentativa pro chamador
    tratar como sempre. Uma exceção de conexão no meio da hibernação é a
    mesma situação, então também conta como motivo pra tentar de novo.
    """
    decorridos = 0
    intervalo = 5
    resposta = None
    while True:
        try:
            resposta = requests.request(metodo, f"{API}{rota}", timeout=TIMEOUT, **kwargs)
        except requests.RequestException:
            resposta = None
        else:
            if resposta.status_code not in (502, 503, 504):
                return resposta
        if decorridos >= 90:
            return resposta
        time.sleep(intervalo)
        decorridos += intervalo


st.set_page_config(page_title="Detector de Discurso de Ódio", page_icon="🛡️")


def detalhe(resposta: requests.Response) -> str:
    """O corpo nem sempre é JSON: um proxy na frente da API (o Render acorda
    de hibernação com uma página de erro em HTML) responde outra coisa."""
    try:
        return resposta.json().get("detail", "Erro inesperado")
    except ValueError:
        return f"A API respondeu {resposta.status_code}. Tente de novo em instantes."


aba_publica, aba_treino = st.tabs(["Classificar", "Painel de treino"])

with aba_publica:
    st.title("Detector de Discurso de Ódio")
    texto = st.text_area("Comentário", max_chars=1000)
    if st.button("Classificar", disabled=not texto.strip()):
        with st.spinner("Classificando — se a API estava hibernando, pode levar até 1 minuto..."):
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
            st.warning(detalhe(resposta))

    # Primeira chamada da aba: se a API (ou o próprio container do Streamlit)
    # estava hibernando, é aqui que a espera de até ~1 minuto acontece. O
    # spinner some sozinho assim que a resposta chega, então em uso normal
    # (API já acordada) ele nem chega a aparecer.
    with st.spinner("Conectando à API — se ela estava hibernando, pode levar até 1 minuto..."):
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
                with st.spinner(
                    "Entrando — se a API estava hibernando, pode levar até 1 minuto..."
                ):
                    resposta = pedir(
                        "POST", "/auth/login", data={"username": email, "password": senha}
                    )
                if resposta is None:
                    pass
                elif resposta.status_code == 200:
                    st.session_state.token = resposta.json()["access_token"]
                    st.rerun()
                elif resposta.status_code == 401:
                    st.error("Email ou senha incorretos.")
                else:
                    # Um 500 ou um 429 não são senha errada; dizer que são
                    # manda o usuário caçar o problema no lugar errado.
                    st.error(detalhe(resposta))
    else:
        cabecalho = {"Authorization": f"Bearer {st.session_state.token}"}

        with st.form("ensinar"):
            novo = st.text_area("Comentário para ensinar", max_chars=1000)
            rotulo = st.radio("É discurso de ódio?", ["Não", "Sim"], horizontal=True)
            if st.form_submit_button("Guardar este exemplo"):
                with st.spinner(
                    "Guardando o exemplo — se a API estava hibernando, pode levar até 1 minuto..."
                ):
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
                    st.error(detalhe(resposta))

        if st.button("Atualizar modelo agora"):
            # O st.status logo abaixo só existe depois que o POST /treinos já
            # respondeu 202; a espera pela hibernação acontece antes disso, então
            # precisa do próprio spinner aqui.
            with st.spinner(
                "Iniciando o treino — se a API estava hibernando, pode levar até 1 minuto..."
            ):
                # F1 "de antes" vem daqui, não do resultado do treino: se ele mantiver o
                # modelo, o que está no ar continua sendo este número.
                antes = pedir("GET", "/metricas")
                f1_antes = None
                if antes is not None and antes.status_code == 200:
                    f1_antes = antes.json()["f1_modelo"]

                resposta = pedir("POST", "/treinos", headers=cabecalho)
            if resposta is None:
                pass
            elif resposta.status_code == 202:
                treino_id = resposta.json()["id"]
                final = None
                # "teto": os 60s passaram sem concluir. "erro": o poll parou antes disso
                # (rede caiu, resposta ruim) — a mensagem de erro já foi mostrada, então o
                # teto não deve ser reclamado por cima de um tempo que não se passou.
                motivo = "teto"
                # Uma rodada real leva ~7s nesta máquina; o teto de 60s é só uma rede de
                # segurança para não travar o painel se algo emperrar.
                with st.status("Treinando o modelo...", expanded=True) as status:
                    inicio = time.time()
                    while time.time() - inicio < 60:
                        acompanha = pedir("GET", f"/treinos/{treino_id}", headers=cabecalho)
                        if acompanha is None:
                            motivo = "erro"
                            break
                        if acompanha.status_code == 401:
                            st.session_state.token = None
                            st.warning("Sessão expirada, entre de novo.")
                            st.rerun()
                        if acompanha.status_code != 200:
                            st.warning(detalhe(acompanha))
                            motivo = "erro"
                            break
                        dado = acompanha.json()
                        if dado["status"] in ("concluido", "falhou"):
                            final = dado
                            status.update(
                                label="Treino concluído"
                                if dado["status"] == "concluido"
                                else "Treino falhou",
                                state="complete" if dado["status"] == "concluido" else "error",
                            )
                            break
                        time.sleep(2)

                if final is not None and final["status"] == "falhou":
                    st.error(f"O treino falhou: {final['erro']}")
                elif final is not None and final["substituiu"] is True:
                    st.success(f"Modelo atualizado! O novo F1 é {final['f1_macro']:.3f}.")
                elif final is not None and final["substituiu"] is False:
                    texto_antes = f"{f1_antes:.3f}" if f1_antes is not None else "—"
                    st.info(
                        f"O treino terminou, mas mantive o modelo que já estava no ar: esta "
                        f"rodada chegou a F1 {final['f1_macro']:.3f}, contra {texto_antes} do "
                        f"modelo atual. Isso é normal — o resultado varia um pouco a cada "
                        f"rodada, e só troco o modelo quando o novo realmente é melhor."
                    )
                elif final is not None:
                    # substituiu veio None: não deveria acontecer (concluido sempre grava
                    # junto), mas se acontecer é melhor dizer "não sei" do que arriscar.
                    st.warning(
                        f"O treino concluiu, mas não consegui confirmar se o modelo foi "
                        f"atualizado. Confira o treino de id {treino_id}."
                    )
                elif motivo == "teto":
                    st.info(
                        f"Ainda está treinando depois de 1 minuto de espera — isso é raro. "
                        f"Confira o resultado daqui a pouco, é o treino de id {treino_id}."
                    )
                # motivo == "erro": a mensagem já apareceu no loop, nada a acrescentar.
            else:
                st.warning(detalhe(resposta))

        exemplos = pedir("GET", "/exemplos?limite=20", headers=cabecalho)
        if exemplos is None:
            pass
        elif exemplos.status_code == 200:
            st.subheader(f"Exemplos ensinados ({exemplos.json()['total']})")
            st.dataframe(exemplos.json()["itens"], use_container_width=True)
        elif exemplos.status_code == 401:
            st.session_state.token = None
            st.warning("Sessão expirada, entre de novo.")
            st.rerun()
        else:
            st.error(detalhe(exemplos))

        if st.button("Sair"):
            st.session_state.token = None
            st.rerun()
