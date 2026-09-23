"""Testa o retry de `pedir()` em frontend/app.py.

frontend/app.py não é um pacote do projeto (é o cliente Streamlit, fora de
app/), então ele é carregado aqui via importlib a partir do caminho do
arquivo, não por um `import frontend.app` normal.

Streamlit roda em "modo nu" fora de `streamlit run`: st.tabs, st.button,
st.spinner etc. viram no-ops (só emitem warning), então o corpo do módulo
executa até o fim sem lançar exceção. A única chamada de rede que dispara
nesse import é o `pedir("GET", "/metricas")` da aba pública — por isso
`requests.request` precisa já estar mockado *antes* do import, senão o teste
tentaria bater em http://localhost:8000 de verdade.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import requests

CAMINHO_APP = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


class _Resposta:
    def __init__(self, status_code, corpo=None):
        self.status_code = status_code
        self._corpo = corpo or {}

    def json(self):
        return self._corpo


@pytest.fixture(scope="module")
def script_html_capturado():
    """Guarda o argumento que o módulo passou pra st.iframe no import — precisa
    ser capturado durante o exec_module, então o patch de st.iframe tem que
    estar de pé antes do import, junto com o de requests.request."""
    return []


@pytest.fixture(scope="module")
def frontend_app(script_html_capturado):
    # O import roda o corpo do módulo até o fim (streamlit em modo nu não
    # levanta), inclusive o `pedir("GET", "/metricas")` da aba pública, que
    # em seguida chama `.json()` na resposta — por isso o corpo precisa das
    # chaves que esse trecho lê.
    corpo_metricas = {"total_predicoes": 0, "taxa_odio": 0.0, "f1_modelo": None}
    original = requests.request
    requests.request = lambda *a, **k: _Resposta(200, corpo_metricas)
    import streamlit as st

    iframe_original = st.iframe
    st.iframe = lambda html, **k: script_html_capturado.append(html)
    try:
        spec = importlib.util.spec_from_file_location("frontend_app_sob_teste", CAMINHO_APP)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
    finally:
        requests.request = original
        st.iframe = iframe_original
    return modulo


@pytest.fixture
def sem_dormir(monkeypatch, frontend_app):
    """time.sleep vira no-op, só pra não perder tempo real nos poucos casos em
    que o laço chega a dormir de verdade. Sozinho isso NÃO evita a espera de
    ~90s nos testes de deadline: o prazo de pedir() é por time.monotonic(),
    que é relógio de parede real se não for mockado também — ver
    `relogio_falso`."""
    chamadas = []
    monkeypatch.setattr(frontend_app.time, "sleep", lambda s: chamadas.append(s))
    return chamadas


@pytest.fixture
def relogio_falso(monkeypatch, frontend_app):
    """Troca time.monotonic() por um relógio controlado à mão (começa em 0,
    só anda quando o teste manda). Os testes que precisam do prazo de 90s se
    esgotar avançam esse relógio dentro do próprio fake de requests.request —
    modelando o tempo que a chamada de rede "gastou" — em vez de depender do
    tempo real passar."""
    estado = {"agora": 0.0}
    monkeypatch.setattr(frontend_app.time, "monotonic", lambda: estado["agora"])
    return estado


def test_200_retorna_na_hora_sem_dormir(frontend_app, sem_dormir, monkeypatch):
    monkeypatch.setattr(frontend_app.requests, "request", lambda *a, **k: _Resposta(200))

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 200
    assert sem_dormir == []


def test_502_depois_200_retorna_o_200(frontend_app, sem_dormir, monkeypatch):
    respostas = iter([_Resposta(502), _Resposta(200)])
    monkeypatch.setattr(frontend_app.requests, "request", lambda *a, **k: next(respostas))

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 200
    assert len(sem_dormir) == 1


def test_502_persistente_retorna_o_ultimo_502_sem_travar(
    frontend_app, sem_dormir, relogio_falso, monkeypatch
):
    def request_falso(*a, **k):
        relogio_falso["agora"] += 10  # cada tentativa "gasta" 10s simulados
        return _Resposta(502)

    monkeypatch.setattr(frontend_app.requests, "request", request_falso)

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 502
    assert len(sem_dormir) > 0


def test_502_persistente_desiste_pelo_prazo_nao_pela_contagem(
    frontend_app, sem_dormir, relogio_falso, monkeypatch
):
    """O bug que motivou o fix: se o tempo gasto dentro de requests.request()
    não contasse pro prazo, um container que aceita a conexão e trava (até
    TIMEOUT=60s por tentativa) faria o teto real virar TIMEOUT vezes o número
    de tentativas — bem mais que os ~90s prometidos no spinner. Aqui cada
    tentativa "demora" 40s simulados, então o prazo de 90s tem que se esgotar
    em 3 tentativas (0s, 40s, 80s — a próxima checagem já está em 120s), não
    nas ~19 que a versão antiga (contagem por número de sleeps) faria."""
    tentativas = []

    def request_falso(*a, **k):
        tentativas.append(1)
        relogio_falso["agora"] += 40
        return _Resposta(502)

    monkeypatch.setattr(frontend_app.requests, "request", request_falso)

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 502
    assert len(tentativas) == 3


def test_script_de_acordar_embute_a_url_da_api_como_literal_js_seguro(
    frontend_app, script_html_capturado
):
    assert len(script_html_capturado) == 1
    script = script_html_capturado[0]
    assert json.dumps(frontend_app.API) in script
    assert 'mode: "no-cors"' in script


def test_excecao_de_conexao_depois_200_retorna_o_200(frontend_app, sem_dormir, monkeypatch):
    tentativas = iter([requests.ConnectionError("recusada"), None])

    def fake_request(*a, **k):
        erro = next(tentativas)
        if erro is not None:
            raise erro
        return _Resposta(200)

    monkeypatch.setattr(frontend_app.requests, "request", fake_request)

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 200
    assert len(sem_dormir) == 1
