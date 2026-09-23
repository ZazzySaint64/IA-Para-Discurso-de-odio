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
def frontend_app():
    # O import roda o corpo do módulo até o fim (streamlit em modo nu não
    # levanta), inclusive o `pedir("GET", "/metricas")` da aba pública, que
    # em seguida chama `.json()` na resposta — por isso o corpo precisa das
    # chaves que esse trecho lê.
    corpo_metricas = {"total_predicoes": 0, "taxa_odio": 0.0, "f1_modelo": None}
    original = requests.request
    requests.request = lambda *a, **k: _Resposta(200, corpo_metricas)
    try:
        spec = importlib.util.spec_from_file_location("frontend_app_sob_teste", CAMINHO_APP)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
    finally:
        requests.request = original
    return modulo


@pytest.fixture
def sem_dormir(monkeypatch, frontend_app):
    """time.sleep vira no-op: como o deadline de pedir() é contado por um
    acumulador incrementado a cada chamada de sleep (não pelo relógio real),
    isso é suficiente para o teste não esperar os ~90s de verdade."""
    chamadas = []
    monkeypatch.setattr(frontend_app.time, "sleep", lambda s: chamadas.append(s))
    return chamadas


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


def test_502_persistente_retorna_o_ultimo_502_sem_travar(frontend_app, sem_dormir, monkeypatch):
    monkeypatch.setattr(frontend_app.requests, "request", lambda *a, **k: _Resposta(502))

    resposta = frontend_app.pedir("GET", "/saude")

    assert resposta.status_code == 502
    assert len(sem_dormir) > 0


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
