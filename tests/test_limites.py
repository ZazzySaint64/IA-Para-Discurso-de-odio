"""O rate limit nunca tinha sido executado: conftest e CI o desligam.

O `Limiter` do slowapi lê `self.enabled` a cada request, então basta ligá-lo
para este teste e zerar o balde — nenhuma gambiarra de reimport, nenhum
código de produção torcido para ser testável.
"""

from fastapi.testclient import TestClient

from app import ml
from app.database import get_db
from app.limites import limiter
from app.main import app


def test_trigesima_primeira_predicao_no_minuto_devolve_429(sessao, monkeypatch):
    monkeypatch.setattr(ml, "prever", lambda texto: (1, 0.87))
    monkeypatch.setattr(ml, "modelo_carregado", lambda: True)
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    app.dependency_overrides[get_db] = lambda: sessao
    try:
        with TestClient(app) as c:
            for _ in range(30):
                assert c.post("/predicoes", json={"texto": "oi"}).status_code == 201
            resposta = c.post("/predicoes", json={"texto": "oi"})
    finally:
        app.dependency_overrides.clear()
        limiter.reset()

    assert resposta.status_code == 429
    assert isinstance(resposta.json()["detail"], str)
    assert resposta.headers["Retry-After"] == "60"
