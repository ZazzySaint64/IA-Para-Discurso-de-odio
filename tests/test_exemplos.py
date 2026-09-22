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


def test_exemplo_duplicado_concorrente_vira_409_e_nao_500(cliente_logado, sessao, monkeypatch):
    """Duas requisições simultâneas passam pela pré-checagem juntas; a perdedora
    bate na UNIQUE do banco. Sem tratar, o handler global devolve 500 justamente
    no caso em que a API promete 409.
    """
    from sqlalchemy.exc import IntegrityError

    commit_real = sessao.commit
    explodiu = []

    def commit_que_falha_uma_vez():
        if not explodiu:
            explodiu.append(True)
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))
        return commit_real()

    monkeypatch.setattr(sessao, "commit", commit_que_falha_uma_vez)

    resposta = cliente_logado.post("/exemplos", json={"texto": "te odeio", "label": 1})
    assert resposta.status_code == 409
    assert "já" in resposta.json()["detail"].lower()
