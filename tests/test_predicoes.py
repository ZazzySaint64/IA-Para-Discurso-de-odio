def test_predicao_caminho_feliz(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "eu te odeio"})
    assert resposta.status_code == 201
    corpo = resposta.json()
    assert corpo["label"] == 1
    assert corpo["rotulo"] == "Discurso de Ódio"
    assert corpo["confianca"] == 0.87


def test_predicao_texto_vazio(cliente):
    resposta = cliente.post("/predicoes", json={"texto": ""})
    assert resposta.status_code == 422
    assert "detail" in resposta.json()


def test_predicao_texto_so_espaco(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "   "})
    assert resposta.status_code == 422


def test_predicao_texto_longo_demais(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "a" * 1001})
    assert resposta.status_code == 422


def test_predicao_sem_modelo_devolve_503(cliente, monkeypatch):
    from app import ml

    def indisponivel(texto):
        raise ml.ModeloIndisponivelError()

    monkeypatch.setattr(ml, "prever", indisponivel)
    resposta = cliente.post("/predicoes", json={"texto": "oi"})
    assert resposta.status_code == 503


def test_predicao_e_gravada_no_banco(cliente, sessao):
    from app.models import Predicao

    cliente.post("/predicoes", json={"texto": "eu te odeio"})
    gravadas = sessao.query(Predicao).all()
    assert len(gravadas) == 1
    assert gravadas[0].texto == "eu te odeio"
    assert gravadas[0].label == 1


def test_erro_inesperado_mantem_o_formato_detail(sessao, monkeypatch):
    """Um commit que falha ainda deve devolver {"detail": "<string>"}.

    `TestClient` normal (a fixture `cliente`) relança exceções do servidor em vez de
    devolver a resposta 500, então este teste monta seu próprio cliente com
    `raise_server_exceptions=False` — a versão instalada do TestClient (starlette
    1.6.0, backend httpx) não aceita esse parâmetro por requisição.
    """
    from fastapi.testclient import TestClient

    from app import ml
    from app.database import get_db
    from app.main import app

    monkeypatch.setattr(ml, "prever", lambda texto: (1, 0.87))
    monkeypatch.setattr(ml, "modelo_carregado", lambda: True)

    def explode():
        raise RuntimeError("banco caiu")

    monkeypatch.setattr(sessao, "commit", explode)

    app.dependency_overrides[get_db] = lambda: sessao
    try:
        with TestClient(app, raise_server_exceptions=False) as cliente_sem_raise:
            resposta = cliente_sem_raise.post("/predicoes", json={"texto": "oi"})
    finally:
        app.dependency_overrides.clear()

    assert resposta.status_code == 500
    assert isinstance(resposta.json()["detail"], str)


def test_listar_predicoes_exige_token(cliente):
    assert cliente.get("/predicoes").status_code == 401


def test_listar_predicoes_pagina(cliente_logado):
    for i in range(3):
        cliente_logado.post("/predicoes", json={"texto": f"frase {i}"})
    resposta = cliente_logado.get("/predicoes?limite=2")
    assert resposta.status_code == 200
    assert resposta.json()["total"] == 3
    assert len(resposta.json()["itens"]) == 2


def test_corpo_que_nao_e_objeto_nao_devolve_detail_comecando_com_dois_pontos(cliente):
    """`loc` == ("body",): não há nome de campo, então o prefixo tem que sumir."""
    resposta = cliente.post("/predicoes", json=["isto não é um objeto"])
    assert resposta.status_code == 422
    detalhe = resposta.json()["detail"]
    assert not detalhe.startswith(":")
    assert detalhe.strip() == detalhe
