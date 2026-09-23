import os

os.environ["RATE_LIMIT_ATIVO"] = "false"
os.environ["JWT_SECRET"] = "segredo-de-teste-com-mais-de-32-bytes-de-tamanho"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import ml
from app.database import Base, get_db
from app.main import app


@pytest.fixture
def sessao():
    """SQLite em memória. Os testes não encostam no Postgres real."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine, expire_on_commit=False)
    db = Sessao()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def cliente(monkeypatch, sessao):
    """Cliente HTTP com o modelo trocado por um fake.

    Teste de rota não pode depender de .pkl nem da acurácia do modelo:
    o que está sendo testado é o contrato HTTP.
    """
    monkeypatch.setattr(ml, "prever", lambda texto: (1, 0.87))
    monkeypatch.setattr(ml, "modelo_carregado", lambda: True)
    app.dependency_overrides[get_db] = lambda: sessao
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def usuario(sessao):
    from app import security
    from app.models import Usuario

    u = Usuario(email="eu@exemplo.com", senha_hash=security.gerar_hash("senha-de-teste"))
    sessao.add(u)
    sessao.commit()
    return u


@pytest.fixture
def cliente_logado(cliente, usuario):
    resposta = cliente.post(
        "/auth/login", data={"username": usuario.email, "password": "senha-de-teste"}
    )
    token = resposta.json()["access_token"]
    cliente.headers["Authorization"] = f"Bearer {token}"
    return cliente
