import os

os.environ["RATE_LIMIT_ATIVO"] = "false"

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
