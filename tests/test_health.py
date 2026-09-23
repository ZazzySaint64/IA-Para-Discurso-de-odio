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


def test_raiz_redireciona_para_docs():
    """Uma API REST não tem homepage: GET / manda o visitante pra /docs
    em vez de devolver 404. Sem depender de modelo/banco carregados."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, follow_redirects=False) as c:
        resposta = c.get("/")

    assert 300 <= resposta.status_code < 400
    assert resposta.headers["location"] == "/docs"
