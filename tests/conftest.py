import pytest
from fastapi.testclient import TestClient

from app import ml
from app.main import app


@pytest.fixture
def cliente(monkeypatch):
    """Cliente HTTP com o modelo trocado por um fake.

    Teste de rota não pode depender de .pkl nem da acurácia do modelo:
    o que está sendo testado é o contrato HTTP.
    """
    monkeypatch.setattr(ml, "prever", lambda texto: (1, 0.87))
    monkeypatch.setattr(ml, "modelo_carregado", lambda: True)
    with TestClient(app) as c:
        yield c
