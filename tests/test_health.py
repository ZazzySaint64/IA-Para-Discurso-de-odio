def test_health_com_modelo_carregado(cliente):
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["modelo"] is True


def test_health_sem_modelo_devolve_503(cliente, monkeypatch):
    from app import ml

    monkeypatch.setattr(ml, "modelo_carregado", lambda: False)
    resposta = cliente.get("/health")
    assert resposta.status_code == 503


def test_health_com_banco_indisponivel_devolve_503(cliente, sessao, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    def falha(*args, **kwargs):
        raise SQLAlchemyError("conexão recusada")

    monkeypatch.setattr(sessao, "execute", falha)
    resposta = cliente.get("/health")
    assert resposta.status_code == 503
    detalhe = resposta.json()["detail"]
    assert isinstance(detalhe, str)
    assert "banco" in detalhe.lower()
