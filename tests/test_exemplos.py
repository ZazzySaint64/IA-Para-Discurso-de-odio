def test_criar_exemplo_exige_token(cliente):
    resposta = cliente.post("/exemplos", json={"texto": "te odeio", "label": 1})
    assert resposta.status_code == 401


def test_criar_exemplo_caminho_feliz(cliente_logado):
    resposta = cliente_logado.post("/exemplos", json={"texto": "te odeio", "label": 1})
    assert resposta.status_code == 201
    assert resposta.json()["texto"] == "te odeio"


def test_criar_exemplo_repetido_devolve_409(cliente_logado):
    cliente_logado.post("/exemplos", json={"texto": "te odeio", "label": 1})
    resposta = cliente_logado.post("/exemplos", json={"texto": "te odeio", "label": 1})
    assert resposta.status_code == 409
    assert "já" in resposta.json()["detail"].lower()


def test_criar_exemplo_com_label_invalido(cliente_logado):
    resposta = cliente_logado.post("/exemplos", json={"texto": "oi", "label": 7})
    assert resposta.status_code == 422


def test_listar_exemplos_pagina(cliente_logado):
    for i in range(5):
        cliente_logado.post("/exemplos", json={"texto": f"frase {i}", "label": 0})
    resposta = cliente_logado.get("/exemplos?limite=2&pular=0")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["total"] == 5
    assert len(corpo["itens"]) == 2


def test_listar_exemplos_exige_token(cliente):
    assert cliente.get("/exemplos").status_code == 401
