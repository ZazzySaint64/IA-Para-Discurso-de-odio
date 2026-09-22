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


def test_exemplo_duplicado_concorrente_vira_409_e_nao_500(
    cliente_logado, sessao, usuario, monkeypatch
):
    """Duas requisições simultâneas passam pela pré-checagem juntas; a perdedora
    bate na UNIQUE do banco. Sem tratar, o handler global devolve 500 justamente
    no caso em que a API promete 409.

    A corrida é simulada gravando a linha concorrente dentro do próprio commit
    do request: o flush leva as duas de uma vez e a UNIQUE estoura de verdade,
    vinda do SQLAlchemy. Um IntegrityError só levantado por um mock não sujaria
    a sessão, e aí o teste não teria como cobrar o rollback.
    """
    from app.models import Exemplo

    commit_real = sessao.commit
    ja_correu = []

    def commit_com_corrida():
        if not ja_correu:
            ja_correu.append(True)
            sessao.add(Exemplo(texto="te odeio", label=0, usuario_id=usuario.id))
        return commit_real()

    monkeypatch.setattr(sessao, "commit", commit_com_corrida)
    resposta = cliente_logado.post("/exemplos", json={"texto": "te odeio", "label": 1})

    assert resposta.status_code == 409
    assert "já" in resposta.json()["detail"].lower()

    # Sem o db.rollback() antes de responder, a sessão fica em
    # PendingRollbackError e a próxima query explode. É esta linha que torna o
    # rollback obrigatório: só o 409 acima passaria sem ela.
    assert cliente_logado.get("/exemplos").status_code == 200
