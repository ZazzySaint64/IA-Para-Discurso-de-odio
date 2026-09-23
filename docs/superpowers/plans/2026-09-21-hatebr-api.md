# HateBR como serviço — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o classificador de discurso de ódio de dois scripts Streamlit em um serviço backend com API REST documentada, Postgres, JWT, Docker, testes, CI e deploy público.

**Architecture:** FastAPI serve predições a partir de um pipeline sklearn carregado uma vez em memória e grava tudo em Postgres via SQLAlchemy. O treino vive em `ml/`, fora do processo da API, porque inferência e treino têm perfis de recurso opostos. O Streamlit vira cliente HTTP.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL 16, PyJWT, bcrypt, slowapi, pytest, ruff, Docker Compose, GitHub Actions, Render.

**Spec:** `docs/superpowers/specs/2026-09-21-hatebr-api-design.md`

## Global Constraints

- Python 3.12 (a máquina de desenvolvimento tem 3.12.10; o Dockerfile usa `python:3.12-slim`).
- SQLAlchemy em estilo 2.0: `DeclarativeBase`, `Mapped`, `mapped_column`. Nunca o estilo 1.x `Column(...)` em classe `Base = declarative_base()`.
- Nomes de domínio em português, como no código existente: `texto`, `label`, `confianca`, `exemplo`, `predicao`, `treino`, `usuario`. Nomes de framework em inglês, como a biblioteca exige.
- `app/` NUNCA importa de `ml/`. Violação disso é erro de revisão, não questão de estilo.
- Toda coluna de data é `DateTime(timezone=True)` com `server_default=func.now()`.
- Enums no banco são `sa.Enum(..., native_enum=False)` (VARCHAR + CHECK), nunca ENUM nativo do Postgres — migrar ENUM nativo é doloroso e não funciona em SQLite.
- Testes usam `fastapi.testclient.TestClient` (síncrono). Sem `pytest-asyncio`.
- Senha tem limite de 72 bytes: é o limite do bcrypt. Validar no schema, não deixar truncar em silêncio.
- Todo commit segue o padrão de mensagem já usado no repo: imperativo, em português, sem prefixo `feat:`/`fix:`.
- Toda mensagem de commit termina com: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Estrutura de arquivos

| Arquivo | Responsabilidade | Task |
|---|---|---|
| `requirements.txt` | dependências de produção | 1 |
| `requirements-dev.txt` | pytest, ruff, pre-commit | 1 |
| `pyproject.toml` | configuração do ruff e do pytest | 1 |
| `.pre-commit-config.yaml` | ruff no commit | 1 |
| `app/ml.py` | carrega o pipeline, expõe `prever()` | 2 |
| `app/schemas.py` | contratos Pydantic de entrada e saída | 3, 5, 7, 8, 10 |
| `app/main.py` | app FastAPI, lifespan, handlers de erro, limiter | 3 |
| `app/routers/predicoes.py` | `POST /predicoes`, `GET /predicoes` | 3, 8 |
| `app/config.py` | `Settings` via pydantic-settings | 4 |
| `app/database.py` | engine, `SessionLocal`, `get_db()` | 4 |
| `app/models.py` | tabelas SQLAlchemy | 4, 5, 7, 10 |
| `alembic/` | migrations | 4 |
| `app/security.py` | bcrypt, JWT, dependência de usuário atual | 5 |
| `app/routers/auth.py` | `POST /auth/login` | 5 |
| `app/seed.py` | cria usuário admin, importa o CSV legado | 6 |
| `app/routers/exemplos.py` | `POST /exemplos`, `GET /exemplos` | 7 |
| `app/routers/metricas.py` | `GET /metricas` | 8 |
| `ml/dados.py` | junta HateBR + ToLD-BR + tabela `exemplo` | 9 |
| `ml/treinar.py` | 5-fold CV, salva um artefato só | 9 |
| `app/routers/treinos.py` | `POST /treinos`, `GET /treinos/{id}` | 10 |
| `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` | containers | 11 |
| `.github/workflows/ci.yml`, `.github/dependabot.yml` | automação | 12 |
| `render.yaml` | deploy | 13 |
| `frontend/app.py` | Streamlit como cliente HTTP | 14 |
| `README.md` | documentação final | 15 |

Arquivos removidos ao longo do plano: `classificador.py` e `painel_treino.py` (Task 14), `testar_modelo.py` (Task 2), `dados_treino.py` e `retreinar_modelo.py` (Task 9), `gerar_hash_senha.py` (Task 6), `test_dados_treino.py` (Task 9), `retreinar.bat` (Task 9).

---

### Task 1: Base do projeto

**Files:**
- Create: `requirements.txt` (substitui o conteúdo atual), `requirements-dev.txt`, `pyproject.toml`, `.pre-commit-config.yaml`, `app/__init__.py`, `tests/__init__.py`, `tests/test_fumaca.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nada
- Produces: ambiente com `pytest` e `ruff` funcionando; pacote `app` importável

- [ ] **Step 1: Confirmar que o Docker está instalado**

Run: `docker compose version`
Expected: imprime a versão, por exemplo `Docker Compose version v2.29.0`. Se der `command not found`, PARE — o Docker Desktop precisa ser instalado e a máquina reiniciada antes de continuar.

- [ ] **Step 2: Criar o ambiente virtual e ativar**

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash no Windows
```

- [ ] **Step 3: Escrever os arquivos de dependências**

`requirements.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.32
sqlalchemy>=2.0
alembic>=1.13
psycopg[binary]>=3.2
pydantic-settings>=2.5
python-multipart>=0.0.12
pyjwt>=2.9
bcrypt>=4.2
slowapi>=0.1.9
joblib>=1.4
scikit-learn>=1.5
pandas>=2.2
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8.3
pytest-cov>=5.0
httpx>=0.27
ruff>=0.7
pre-commit>=4.0
```

- [ ] **Step 4: Escrever o `pyproject.toml`**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 5: Escrever o `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 6: Instalar tudo**

```bash
pip install -r requirements-dev.txt
pre-commit install
```

- [ ] **Step 7: Escrever o teste de fumaça**

`tests/test_fumaca.py`:
```python
def test_pacote_app_importavel():
    import app

    assert app is not None
```

Criar `app/__init__.py` e `tests/__init__.py` vazios.

- [ ] **Step 8: Rodar os testes**

Run: `pytest`
Expected: PASS, 1 teste.

- [ ] **Step 9: Acrescentar ao `.gitignore`**

```
.venv/
.pytest_cache/
.ruff_cache/
htmlcov/
.coverage
```

- [ ] **Step 10: Commit**

```bash
git add requirements.txt requirements-dev.txt pyproject.toml .pre-commit-config.yaml app/ tests/ .gitignore
git commit -m "$(cat <<'EOF'
Prepara o esqueleto do projeto com pytest, ruff e pre-commit

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Camada de modelo (`app/ml.py`)

O `testar_modelo.py` atual carrega dois `.pkl` no momento do import. Aqui isso vira
carregamento explícito, com um artefato só, e com erro claro quando o arquivo não existe.

**Files:**
- Create: `app/ml.py`, `tests/test_ml.py`
- Delete: `testar_modelo.py`

**Interfaces:**
- Consumes: nada
- Produces:
  - `carregar_modelo(caminho: Path) -> None` — carrega o pipeline no módulo
  - `prever(texto: str) -> tuple[int, float]` — devolve `(label, confianca)`
  - `modelo_carregado() -> bool`
  - `ModeloIndisponivelError` — exceção levantada por `prever()` sem modelo carregado

- [ ] **Step 1: Escrever os testes primeiro**

`tests/test_ml.py`:
```python
import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import ml


@pytest.fixture
def caminho_modelo(tmp_path):
    """Pipeline minúsculo, treinado na hora. Não depende do .pkl real."""
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]
    )
    pipeline.fit(
        ["eu te odeio seu lixo", "vá morrer", "bom dia pessoal", "que dia lindo"],
        [1, 1, 0, 0],
    )
    caminho = tmp_path / "modelo.pkl"
    joblib.dump(pipeline, caminho)
    return caminho


def test_prever_sem_modelo_carregado_levanta_erro():
    ml._pipeline = None
    with pytest.raises(ml.ModeloIndisponivelError):
        ml.prever("qualquer coisa")


def test_carregar_modelo_inexistente_levanta_erro(tmp_path):
    with pytest.raises(FileNotFoundError):
        ml.carregar_modelo(tmp_path / "nao_existe.pkl")


def test_prever_devolve_label_e_confianca(caminho_modelo):
    ml.carregar_modelo(caminho_modelo)
    label, confianca = ml.prever("eu te odeio seu lixo")
    assert label in (0, 1)
    assert 0.0 <= confianca <= 1.0


def test_modelo_carregado_reflete_o_estado(caminho_modelo):
    ml._pipeline = None
    assert ml.modelo_carregado() is False
    ml.carregar_modelo(caminho_modelo)
    assert ml.modelo_carregado() is True
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `pytest tests/test_ml.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'app.ml'`

- [ ] **Step 3: Implementar `app/ml.py`**

```python
"""Carrega o pipeline treinado e responde predições.

Este módulo NÃO treina nada. Treino é responsabilidade de `ml/treinar.py`,
que roda fora do processo da API.
"""

from pathlib import Path

import joblib

_pipeline = None


class ModeloIndisponivelError(RuntimeError):
    """Predição pedida antes de o modelo ser carregado."""


def carregar_modelo(caminho: Path) -> None:
    global _pipeline
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}")
    _pipeline = joblib.load(caminho)


def modelo_carregado() -> bool:
    return _pipeline is not None


def prever(texto: str) -> tuple[int, float]:
    """Classifica um texto. Devolve (label, confianca)."""
    if _pipeline is None:
        raise ModeloIndisponivelError("Modelo não carregado")
    label = int(_pipeline.predict([texto])[0])
    confianca = float(_pipeline.predict_proba([texto])[0].max())
    return label, confianca
```

- [ ] **Step 4: Rodar os testes**

Run: `pytest tests/test_ml.py -v`
Expected: PASS, 4 testes.

- [ ] **Step 5: Remover o `testar_modelo.py`**

```bash
git rm testar_modelo.py
```

- [ ] **Step 6: Commit**

```bash
git add app/ml.py tests/test_ml.py
git commit -m "$(cat <<'EOF'
Troca o carregamento implícito do modelo por app/ml.py

O testar_modelo.py carregava dois .pkl como efeito colateral de import.
Agora o carregamento é explícito, de um artefato só, e prever() falha
com erro nomeado quando o modelo não está carregado.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: API base, `/health` e `POST /predicoes`

Sem banco ainda. A predição responde, mas não é gravada — isso entra na Task 4.

**Files:**
- Create: `app/main.py`, `app/schemas.py`, `app/routers/__init__.py`, `app/routers/predicoes.py`, `tests/conftest.py`, `tests/test_predicoes.py`, `tests/test_health.py`

**Interfaces:**
- Consumes: `app.ml.prever`, `app.ml.modelo_carregado`, `app.ml.ModeloIndisponivelError`
- Produces:
  - `app.main.app` — instância FastAPI
  - `PredicaoEntrada(texto: str)` e `PredicaoSaida(label: int, rotulo: str, confianca: float)`
  - fixture `cliente` em `tests/conftest.py`

- [ ] **Step 1: Escrever os schemas**

`app/schemas.py`:
```python
from pydantic import BaseModel, Field, field_validator


class PredicaoEntrada(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)

    @field_validator("texto")
    @classmethod
    def sem_espaco_sobrando(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("texto não pode ser só espaço em branco")
        return v


class PredicaoSaida(BaseModel):
    label: int
    rotulo: str
    confianca: float
```

- [ ] **Step 2: Escrever a fixture de teste**

`tests/conftest.py`:
```python
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
```

- [ ] **Step 3: Escrever os testes**

`tests/test_predicoes.py`:
```python
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
```

`tests/test_health.py`:
```python
def test_health_com_modelo_carregado(cliente):
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["modelo"] is True


def test_health_sem_modelo_devolve_503(cliente, monkeypatch):
    from app import ml

    monkeypatch.setattr(ml, "modelo_carregado", lambda: False)
    resposta = cliente.get("/health")
    assert resposta.status_code == 503
```

- [ ] **Step 4: Rodar os testes e confirmar que falham**

Run: `pytest tests/test_predicoes.py tests/test_health.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 5: Escrever o router de predições**

`app/routers/predicoes.py`:
```python
from fastapi import APIRouter, HTTPException, status

from app import ml
from app.schemas import PredicaoEntrada, PredicaoSaida

router = APIRouter(tags=["predicoes"])

ROTULOS = {0: "Não é Discurso de Ódio", 1: "Discurso de Ódio"}


@router.post("/predicoes", response_model=PredicaoSaida, status_code=status.HTTP_201_CREATED)
def criar_predicao(entrada: PredicaoEntrada) -> PredicaoSaida:
    try:
        label, confianca = ml.prever(entrada.texto)
    except ml.ModeloIndisponivelError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível no momento",
        ) from None
    return PredicaoSaida(label=label, rotulo=ROTULOS[label], confianca=confianca)
```

- [ ] **Step 6: Escrever o `app/main.py`**

O `exception_handler` de `RequestValidationError` existe porque o Pydantic devolve
`detail` como lista de objetos, e o resto da API devolve `detail` como string. Sem
ele, o cliente teria que lidar com dois formatos.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import ml
from app.routers import predicoes

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O modelo é carregado uma vez por processo, não a cada request.
    # Em Task 4 o caminho passa a vir de Settings.
    from pathlib import Path

    caminho = Path("ml/artefatos/modelo.pkl")
    if caminho.exists():
        ml.carregar_modelo(caminho)
    yield


app = FastAPI(
    title="API de Detecção de Discurso de Ódio",
    description="Classificador de comentários em português (TF-IDF + Regressão Logística).",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RequestValidationError)
async def erro_de_validacao(request: Request, exc: RequestValidationError):
    """Padroniza o 422 do Pydantic no mesmo formato dos outros erros."""
    primeiro = exc.errors()[0]
    campo = ".".join(str(p) for p in primeiro["loc"] if p != "body")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": f"{campo}: {primeiro['msg']}"},
    )


@app.exception_handler(RateLimitExceeded)
async def erro_de_limite(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Muitas requisições. Tente de novo em instantes."},
    )


@app.get("/health", tags=["infra"])
def health():
    if not ml.modelo_carregado():
        return JSONResponse(status_code=503, content={"detail": "Modelo não carregado"})
    return {"status": "ok", "modelo": True}


app.include_router(predicoes.router)
```

- [ ] **Step 7: Rodar os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 8: Subir e olhar o Swagger**

Run: `uvicorn app.main:app --reload`
Abrir `http://127.0.0.1:8000/docs`, executar o `POST /predicoes` pelo botão "Try it out".
Expected: `503`, porque ainda não existe `ml/artefatos/modelo.pkl`. Isso é o esperado nesta fase.

- [ ] **Step 9: Commit**

```bash
git add app/ tests/
git commit -m "$(cat <<'EOF'
Cria a API com /health e POST /predicoes

Sem persistência ainda. Inclui o handler que padroniza o 422 do Pydantic
no mesmo formato {"detail": "..."} dos demais erros.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Banco de dados e persistência das predições

**Files:**
- Create: `app/config.py`, `app/database.py`, `app/models.py`, `alembic.ini`, `alembic/env.py`, `alembic/versions/<hash>_inicial.py`, `.env.exemplo`
- Modify: `app/main.py`, `app/routers/predicoes.py`, `tests/conftest.py`
- Test: `tests/test_predicoes.py`

**Interfaces:**
- Consumes: `app.schemas.PredicaoEntrada`
- Produces:
  - `app.config.settings` — objeto `Settings` com `DATABASE_URL`, `JWT_SECRET`, `MODELO_PATH`, `TREINO_HABILITADO`, `RATE_LIMIT_ATIVO`
  - `app.database.Base`, `app.database.get_db`
  - `app.models.Predicao`
  - fixture `sessao` em `tests/conftest.py`

- [ ] **Step 1: Escrever o `app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://hatebr:hatebr@localhost:5432/hatebr"
    JWT_SECRET: str = "troque-isso-em-producao"
    JWT_EXPIRA_MINUTOS: int = 60
    MODELO_PATH: str = "ml/artefatos/modelo.pkl"
    TREINO_HABILITADO: bool = True
    RATE_LIMIT_ATIVO: bool = True


settings = Settings()
```

`.env.exemplo`:
```
DATABASE_URL=postgresql+psycopg://hatebr:hatebr@localhost:5432/hatebr
JWT_SECRET=gere-um-com-python-c-import-secrets-print-secrets-token-urlsafe-32
MODELO_PATH=ml/artefatos/modelo.pkl
TREINO_HABILITADO=true
```

- [ ] **Step 2: Escrever o `app/database.py`**

```python
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 3: Escrever o `app/models.py` com a tabela `predicao`**

```python
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Predicao(Base):
    __tablename__ = "predicao"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    texto: Mapped[str] = mapped_column(String(1000), nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    confianca: Mapped[float] = mapped_column(Float, nullable=False)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

- [ ] **Step 4: Subir o Postgres**

Criar um `docker-compose.yml` provisório com só o banco (a API entra na Task 11):

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: hatebr
      POSTGRES_PASSWORD: hatebr
      POSTGRES_DB: hatebr
    ports:
      - "5432:5432"
    volumes:
      - dados_postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hatebr"]
      interval: 5s
      retries: 5

volumes:
  dados_postgres:
```

Run: `docker compose up -d db`
Expected: container `db` saudável em `docker compose ps`.

- [ ] **Step 5: Inicializar o Alembic**

Template síncrono, porque o projeto usa `Session` síncrona:
```bash
alembic init alembic
```

Editar `alembic/env.py`, trocando o bloco de configuração por:
```python
from app.config import settings
from app.database import Base
from app import models  # noqa: F401  (importa para registrar as tabelas)

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
target_metadata = Base.metadata
```

- [ ] **Step 6: Gerar e aplicar a migration**

```bash
alembic revision --autogenerate -m "cria tabela predicao"
alembic upgrade head
```
Expected: tabela `predicao` existe. Conferir com:
```bash
docker compose exec db psql -U hatebr -d hatebr -c "\d predicao"
```

- [ ] **Step 7: Escrever o teste de persistência**

Acrescentar a `tests/conftest.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db


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
```

E trocar a fixture `cliente` para injetar essa sessão:
```python
@pytest.fixture
def cliente(monkeypatch, sessao):
    monkeypatch.setattr(ml, "prever", lambda texto: (1, 0.87))
    monkeypatch.setattr(ml, "modelo_carregado", lambda: True)
    app.dependency_overrides[get_db] = lambda: sessao
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

Acrescentar a `tests/test_predicoes.py`:
```python
def test_predicao_e_gravada_no_banco(cliente, sessao):
    from app.models import Predicao

    cliente.post("/predicoes", json={"texto": "eu te odeio"})
    gravadas = sessao.query(Predicao).all()
    assert len(gravadas) == 1
    assert gravadas[0].texto == "eu te odeio"
    assert gravadas[0].label == 1
```

- [ ] **Step 8: Rodar o teste e confirmar que falha**

Run: `pytest tests/test_predicoes.py::test_predicao_e_gravada_no_banco -v`
Expected: FAIL, `assert 0 == 1`

- [ ] **Step 9: Gravar a predição no router**

`app/routers/predicoes.py`, função `criar_predicao`:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import ml
from app.database import get_db
from app.models import Predicao
from app.schemas import PredicaoEntrada, PredicaoSaida


@router.post("/predicoes", response_model=PredicaoSaida, status_code=status.HTTP_201_CREATED)
def criar_predicao(entrada: PredicaoEntrada, db: Session = Depends(get_db)) -> PredicaoSaida:
    try:
        label, confianca = ml.prever(entrada.texto)
    except ml.ModeloIndisponivelError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível no momento",
        ) from None

    db.add(Predicao(texto=entrada.texto, label=label, confianca=confianca))
    db.commit()
    return PredicaoSaida(label=label, rotulo=ROTULOS[label], confianca=confianca)
```

- [ ] **Step 10: Ligar o rate limit e o `MODELO_PATH` das settings**

O limiter vive em módulo próprio, `app/limites.py`, e não em `app/main.py`. Se
morasse em `main.py`, os routers precisariam importar de `main.py`, que importa os
routers: import circular.

`app/limites.py`:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address, enabled=settings.RATE_LIMIT_ATIVO)
```

Em `app/main.py`: trocar a criação do limiter por `from app.limites import limiter`, e
trocar o caminho fixo do lifespan por `settings.MODELO_PATH`:
```python
from pathlib import Path

from app.config import settings
from app.limites import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    caminho = Path(settings.MODELO_PATH)
    if caminho.exists():
        ml.carregar_modelo(caminho)
    yield
```

Em `app/routers/predicoes.py`, o decorator exige um parâmetro `request: Request` na
assinatura — é dele que o slowapi tira o IP:
```python
from fastapi import Request

from app.limites import limiter


@router.post("/predicoes", response_model=PredicaoSaida, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
def criar_predicao(
    request: Request, entrada: PredicaoEntrada, db: Session = Depends(get_db)
) -> PredicaoSaida:
    ...
```

O rate limit fica desligado nos testes. Primeira linha de `tests/conftest.py`, antes
de qualquer import de `app`:
```python
import os

os.environ["RATE_LIMIT_ATIVO"] = "false"
```

- [ ] **Step 11: Rodar todos os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 12: Commit**

```bash
git add app/ alembic/ alembic.ini tests/ docker-compose.yml .env.exemplo
git commit -m "$(cat <<'EOF'
Adiciona Postgres, SQLAlchemy e Alembic, e grava cada predição

Toda predição agora vira linha na tabela predicao. Os testes rodam em
SQLite em memória com dependency_overrides, sem encostar no Postgres.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Autenticação com JWT

**Files:**
- Create: `app/security.py`, `app/routers/auth.py`, `tests/test_auth.py`
- Modify: `app/models.py`, `app/schemas.py`, `app/main.py`, `tests/conftest.py`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `app.database.get_db`, `app.models.Base`
- Produces:
  - `app.models.Usuario`
  - `gerar_hash(senha: str) -> str`, `conferir_senha(senha: str, hash_: str) -> bool`
  - `criar_token(usuario_id: int) -> str`
  - `usuario_atual(...) -> Usuario` — dependência FastAPI usada por todas as rotas protegidas
  - fixture `usuario` e fixture `cliente_logado` em `tests/conftest.py`

- [ ] **Step 1: Acrescentar a tabela `usuario` em `app/models.py`**

```python
class Usuario(Base):
    __tablename__ = "usuario"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    senha_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

- [ ] **Step 2: Escrever os testes de segurança e de login**

`tests/test_auth.py`:
```python
from app import security


def test_hash_nao_guarda_a_senha_em_texto():
    hash_ = security.gerar_hash("minha-senha-secreta")
    assert "minha-senha-secreta" not in hash_
    assert hash_.startswith("$2b$")


def test_hashes_da_mesma_senha_sao_diferentes():
    """bcrypt embute salt: duas chamadas com a mesma senha dão hashes distintos."""
    assert security.gerar_hash("abc123") != security.gerar_hash("abc123")


def test_conferir_senha_certa_e_errada():
    hash_ = security.gerar_hash("abc123")
    assert security.conferir_senha("abc123", hash_) is True
    assert security.conferir_senha("abc124", hash_) is False


def test_token_carrega_o_id_do_usuario():
    token = security.criar_token(42)
    assert security.ler_token(token) == 42


def test_token_invalido_devolve_none():
    assert security.ler_token("token.falsificado.aqui") is None


def test_login_com_credencial_certa(cliente, usuario):
    resposta = cliente.post(
        "/auth/login", data={"username": usuario.email, "password": "senha-de-teste"}
    )
    assert resposta.status_code == 200
    assert "access_token" in resposta.json()


def test_login_com_senha_errada(cliente, usuario):
    resposta = cliente.post(
        "/auth/login", data={"username": usuario.email, "password": "errada"}
    )
    assert resposta.status_code == 401


def test_login_com_email_inexistente(cliente):
    resposta = cliente.post(
        "/auth/login", data={"username": "ninguem@exemplo.com", "password": "x"}
    )
    assert resposta.status_code == 401
```

- [ ] **Step 3: Rodar e confirmar que falham**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'app.security'`

- [ ] **Step 4: Escrever o `app/security.py`**

```python
"""Hash de senha e tokens JWT.

bcrypt em vez de SHA-256: SHA-256 é rápido de propósito, o que é exatamente o
que não se quer em senha. bcrypt é lento de propósito e já embute salt, então
a mesma senha gera hashes diferentes e rainbow table não serve.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Usuario

oauth2 = OAuth2PasswordBearer(tokenUrl="auth/login")

LIMITE_BCRYPT_BYTES = 72


def gerar_hash(senha: str) -> str:
    return bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def conferir_senha(senha: str, hash_: str) -> bool:
    return bcrypt.checkpw(senha.encode("utf-8"), hash_.encode("utf-8"))


def criar_token(usuario_id: int) -> str:
    expira = datetime.now(UTC) + timedelta(minutes=settings.JWT_EXPIRA_MINUTOS)
    return jwt.encode(
        {"sub": str(usuario_id), "exp": expira}, settings.JWT_SECRET, algorithm="HS256"
    )


def ler_token(token: str) -> int | None:
    try:
        dados = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return int(dados["sub"])


def usuario_atual(
    token: str = Depends(oauth2), db: Session = Depends(get_db)
) -> Usuario:
    nao_autorizado = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    usuario_id = ler_token(token)
    if usuario_id is None:
        raise nao_autorizado
    usuario = db.get(Usuario, usuario_id)
    if usuario is None:
        raise nao_autorizado
    return usuario
```

- [ ] **Step 5: Escrever o `app/routers/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import security
from app.database import get_db
from app.models import Usuario
from app.schemas import Token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
) -> Token:
    usuario = db.scalar(select(Usuario).where(Usuario.email == form.username))
    if usuario is None or not security.conferir_senha(form.password, usuario.senha_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=security.criar_token(usuario.id), token_type="bearer")
```

Acrescentar a `app/schemas.py`:
```python
class Token(BaseModel):
    access_token: str
    token_type: str
```

Registrar em `app/main.py`: `app.include_router(auth.router)`.

- [ ] **Step 6: Acrescentar as fixtures de usuário**

Em `tests/conftest.py`:
```python
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
```

- [ ] **Step 7: Rodar os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 8: Gerar a migration**

```bash
alembic revision --autogenerate -m "cria tabela usuario"
alembic upgrade head
```

- [ ] **Step 9: Commit**

```bash
git add app/ alembic/versions/ tests/
git commit -m "$(cat <<'EOF'
Adiciona autenticação com bcrypt e JWT

Substitui o SHA-256 em variável de ambiente por bcrypt com salt na tabela
usuario. OAuth2PasswordBearer habilita o botão Authorize no Swagger.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `app/seed.py` e importação do CSV legado

**Files:**
- Create: `app/seed.py`, `tests/test_seed.py`
- Modify: `app/models.py` (tabela `exemplo`)
- Delete: `gerar_hash_senha.py`

**Interfaces:**
- Consumes: `app.security.gerar_hash`, `app.models.Usuario`
- Produces:
  - `app.models.Exemplo`
  - `criar_usuario(db, email, senha) -> Usuario`
  - `gerar_senha() -> str`
  - `importar_csv(db, caminho, usuario_id) -> int` — devolve quantos exemplos importou

- [ ] **Step 1: Acrescentar a tabela `exemplo` em `app/models.py`**

```python
from sqlalchemy import ForeignKey


class Exemplo(Base):
    __tablename__ = "exemplo"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    texto: Mapped[str] = mapped_column(String(1000), unique=True, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuario.id"), nullable=False)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

- [ ] **Step 2: Escrever os testes**

`tests/test_seed.py`:
```python
import pytest

from app import seed
from app.models import Exemplo, Usuario


def test_senha_gerada_tem_entropia_suficiente():
    senha = seed.gerar_senha()
    assert len(senha) >= 20
    assert seed.gerar_senha() != seed.gerar_senha()


def test_criar_usuario_guarda_hash_nao_a_senha(sessao):
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "minha-senha")
    assert usuario.senha_hash != "minha-senha"
    assert sessao.get(Usuario, usuario.id) is not None


def test_criar_usuario_repetido_levanta_erro(sessao):
    seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    with pytest.raises(seed.UsuarioJaExisteError):
        seed.criar_usuario(sessao, "eu@exemplo.com", "abc")


def test_importar_csv_traz_os_exemplos(sessao, tmp_path):
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text(
        "comentario,label_final\nte odeio,1\nbom dia,0\n", encoding="utf-8"
    )
    quantos = seed.importar_csv(sessao, csv, usuario.id)
    assert quantos == 2
    assert sessao.query(Exemplo).count() == 2


def test_importar_csv_ignora_texto_repetido(sessao, tmp_path):
    """A constraint UNIQUE existe justamente porque o CSV atual aceita repetido."""
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text(
        "comentario,label_final\nte odeio,1\nte odeio,1\n", encoding="utf-8"
    )
    quantos = seed.importar_csv(sessao, csv, usuario.id)
    assert quantos == 1
```

- [ ] **Step 3: Rodar e confirmar que falham**

Run: `pytest tests/test_seed.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'app.seed'`

- [ ] **Step 4: Escrever o `app/seed.py`**

```python
"""Cria o usuário administrador e importa o feedback legado.

Substitui o gerar_hash_senha.py: a senha nunca vira variável de ambiente,
e com --gerar-senha ela nem chega a ser digitada.

Uso:
    python -m app.seed --gerar-senha
    python -m app.seed --email eu@exemplo.com
    python -m app.seed --resetar-senha --email eu@exemplo.com
    python -m app.seed --importar-csv feedback_treino.csv
"""

import argparse
import csv as csv_lib
import secrets
from getpass import getpass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import security
from app.database import SessionLocal
from app.models import Exemplo, Usuario


class UsuarioJaExisteError(RuntimeError):
    pass


def gerar_senha() -> str:
    """~120 bits de entropia, via CSPRNG do sistema operacional."""
    return secrets.token_urlsafe(15)


def criar_usuario(db: Session, email: str, senha: str) -> Usuario:
    if db.scalar(select(Usuario).where(Usuario.email == email)) is not None:
        raise UsuarioJaExisteError(f"Já existe usuário com o email {email}")
    usuario = Usuario(email=email, senha_hash=security.gerar_hash(senha))
    db.add(usuario)
    db.commit()
    return usuario


def resetar_senha(db: Session, email: str, senha: str) -> Usuario:
    usuario = db.scalar(select(Usuario).where(Usuario.email == email))
    if usuario is None:
        raise ValueError(f"Não existe usuário com o email {email}")
    usuario.senha_hash = security.gerar_hash(senha)
    db.commit()
    return usuario


def importar_csv(db: Session, caminho: Path, usuario_id: int) -> int:
    """Importa feedback_treino.csv para a tabela exemplo, pulando repetidos."""
    ja_existem = {t for (t,) in db.execute(select(Exemplo.texto))}
    importados = 0
    with open(caminho, encoding="utf-8") as f:
        for linha in csv_lib.DictReader(f):
            texto = linha["comentario"].strip()
            if not texto or texto in ja_existem:
                continue
            db.add(
                Exemplo(texto=texto, label=int(linha["label_final"]), usuario_id=usuario_id)
            )
            ja_existem.add(texto)
            importados += 1
    db.commit()
    return importados


def main() -> None:
    p = argparse.ArgumentParser(description="Cria o usuário admin e importa dados legados.")
    p.add_argument("--email", default="admin@hatebr.local")
    p.add_argument("--gerar-senha", action="store_true", help="gera uma senha forte")
    p.add_argument("--resetar-senha", action="store_true")
    p.add_argument("--importar-csv", type=Path, help="caminho do feedback_treino.csv")
    args = p.parse_args()

    db = SessionLocal()
    try:
        if args.gerar_senha:
            senha = gerar_senha()
            mostrar = True
        else:
            senha = getpass("Senha (não aparece enquanto você digita): ")
            mostrar = False

        if args.resetar_senha:
            usuario = resetar_senha(db, args.email, senha)
            print(f"Senha trocada para {usuario.email}")
        else:
            usuario = criar_usuario(db, args.email, senha)
            print(f"Usuário criado: {usuario.email}")

        if mostrar:
            print(f"Senha (aparece só esta vez, salve agora): {senha}")

        if args.importar_csv:
            quantos = importar_csv(db, args.importar_csv, usuario.id)
            print(f"{quantos} exemplos importados de {args.importar_csv}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Rodar os testes**

Run: `pytest tests/test_seed.py -v`
Expected: PASS, 5 testes.

- [ ] **Step 6: Gerar a migration e rodar de verdade**

```bash
alembic revision --autogenerate -m "cria tabela exemplo"
alembic upgrade head
python -m app.seed --gerar-senha --email <seu email> --importar-csv feedback_treino.csv
```
Expected: imprime o usuário criado, a senha uma única vez, e quantos exemplos vieram
do CSV. **Salvar a senha no gerenciador de senhas agora** — ela não é recuperável.

- [ ] **Step 7: Remover o `gerar_hash_senha.py`**

```bash
git rm gerar_hash_senha.py
```

- [ ] **Step 8: Commit**

```bash
git add app/ alembic/versions/ tests/
git commit -m "$(cat <<'EOF'
Adiciona app/seed.py e importa o feedback_treino.csv para o banco

Substitui o gerar_hash_senha.py. A senha pode ser gerada com
secrets.token_urlsafe, aparece uma vez no terminal e só o hash é gravado.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `POST /exemplos` e `GET /exemplos`

**Files:**
- Create: `app/routers/exemplos.py`, `tests/test_exemplos.py`
- Modify: `app/schemas.py`, `app/main.py`

**Interfaces:**
- Consumes: `app.security.usuario_atual`, `app.models.Exemplo`
- Produces:
  - `ExemploEntrada(texto: str, label: Literal[0, 1])`
  - `ExemploSaida(id, texto, label, criado_em)`
  - `Pagina[T]` — envelope de paginação com `itens` e `total`

- [ ] **Step 1: Escrever os testes**

`tests/test_exemplos.py`:
```python
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
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `pytest tests/test_exemplos.py -v`
Expected: FAIL, `404` em vez de `401`/`201` — as rotas não existem.

- [ ] **Step 3: Acrescentar os schemas**

Em `app/schemas.py`:
```python
from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ExemploEntrada(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)
    label: Literal[0, 1]

    @field_validator("texto")
    @classmethod
    def sem_espaco_sobrando(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("texto não pode ser só espaço em branco")
        return v


class ExemploSaida(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    texto: str
    label: int
    criado_em: datetime


class Pagina(BaseModel, Generic[T]):
    total: int
    itens: list[T]
```

- [ ] **Step 4: Escrever o router**

`app/routers/exemplos.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Exemplo, Usuario
from app.schemas import ExemploEntrada, ExemploSaida, Pagina
from app.security import usuario_atual

router = APIRouter(prefix="/exemplos", tags=["exemplos"])


@router.post("", response_model=ExemploSaida, status_code=status.HTTP_201_CREATED)
def criar_exemplo(
    entrada: ExemploEntrada,
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> Exemplo:
    existente = db.scalar(select(Exemplo).where(Exemplo.texto == entrada.texto))
    if existente is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Esse texto já foi ensinado ao modelo",
        )
    exemplo = Exemplo(texto=entrada.texto, label=entrada.label, usuario_id=usuario.id)
    db.add(exemplo)
    db.commit()
    return exemplo


@router.get("", response_model=Pagina[ExemploSaida])
def listar_exemplos(
    limite: int = Query(50, ge=1, le=100),
    pular: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> Pagina[ExemploSaida]:
    total = db.scalar(select(func.count()).select_from(Exemplo))
    itens = db.scalars(
        select(Exemplo).order_by(Exemplo.criado_em.desc()).limit(limite).offset(pular)
    ).all()
    return Pagina(total=total, itens=[ExemploSaida.model_validate(i) for i in itens])
```

Registrar em `app/main.py`: `app.include_router(exemplos.router)`.

- [ ] **Step 5: Rodar os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 6: Commit**

```bash
git add app/ tests/
git commit -m "$(cat <<'EOF'
Adiciona as rotas de exemplos, com 409 em texto repetido

A constraint UNIQUE na tabela substitui o feedback_treino.csv, que
aceitava a mesma frase várias vezes e enviesava o treino.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `GET /predicoes` e `GET /metricas`

**Files:**
- Create: `app/routers/metricas.py`, `tests/test_metricas.py`
- Modify: `app/routers/predicoes.py`, `app/schemas.py`, `app/main.py`, `tests/test_predicoes.py`

**Interfaces:**
- Consumes: `app.models.Predicao`, `app.models.Exemplo`, `app.security.usuario_atual`
- Produces:
  - `PredicaoRegistro(id, texto, label, confianca, criado_em)`
  - `Metricas(total_predicoes, taxa_odio, total_exemplos, f1_modelo, ultimo_treino_em)`

- [ ] **Step 1: Escrever os testes**

`tests/test_metricas.py`:
```python
def test_metricas_com_banco_vazio(cliente):
    resposta = cliente.get("/metricas")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["total_predicoes"] == 0
    assert corpo["taxa_odio"] == 0.0
    assert corpo["total_exemplos"] == 0


def test_metricas_conta_predicoes_e_taxa(cliente):
    for _ in range(4):
        cliente.post("/predicoes", json={"texto": "eu te odeio"})
    resposta = cliente.get("/metricas")
    corpo = resposta.json()
    assert corpo["total_predicoes"] == 4
    # a fixture `cliente` faz prever() devolver sempre label 1
    assert corpo["taxa_odio"] == 1.0


def test_metricas_nao_exige_token(cliente):
    assert cliente.get("/metricas").status_code == 200
```

Acrescentar a `tests/test_predicoes.py`:
```python
def test_listar_predicoes_exige_token(cliente):
    assert cliente.get("/predicoes").status_code == 401


def test_listar_predicoes_pagina(cliente_logado):
    for i in range(3):
        cliente_logado.post("/predicoes", json={"texto": f"frase {i}"})
    resposta = cliente_logado.get("/predicoes?limite=2")
    assert resposta.status_code == 200
    assert resposta.json()["total"] == 3
    assert len(resposta.json()["itens"]) == 2
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `pytest tests/test_metricas.py -v`
Expected: FAIL, `404` — a rota não existe.

- [ ] **Step 3: Acrescentar os schemas**

Em `app/schemas.py`:
```python
class PredicaoRegistro(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    texto: str
    label: int
    confianca: float
    criado_em: datetime


class Metricas(BaseModel):
    total_predicoes: int
    taxa_odio: float
    total_exemplos: int
    f1_modelo: float | None
    ultimo_treino_em: datetime | None
```

- [ ] **Step 4: Acrescentar `GET /predicoes`**

Em `app/routers/predicoes.py`, acrescentar aos imports do topo:
```python
from fastapi import Query
from sqlalchemy import func, select

from app.models import Usuario
from app.schemas import Pagina, PredicaoRegistro
from app.security import usuario_atual
```

E a rota:
```python
@router.get("/predicoes", response_model=Pagina[PredicaoRegistro])
def listar_predicoes(
    limite: int = Query(50, ge=1, le=100),
    pular: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> Pagina[PredicaoRegistro]:
    total = db.scalar(select(func.count()).select_from(Predicao))
    itens = db.scalars(
        select(Predicao).order_by(Predicao.criado_em.desc()).limit(limite).offset(pular)
    ).all()
    return Pagina(total=total, itens=[PredicaoRegistro.model_validate(i) for i in itens])
```

- [ ] **Step 5: Escrever o `app/routers/metricas.py`**

O campo `f1_modelo` fica `None` até a Task 10 criar a tabela `treino`; depois disso
ele passa a ler de lá. Nesta task, devolver `None` é o comportamento correto.

```python
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Exemplo, Predicao
from app.schemas import Metricas

router = APIRouter(tags=["metricas"])


@router.get("/metricas", response_model=Metricas)
def obter_metricas(db: Session = Depends(get_db)) -> Metricas:
    total = db.scalar(select(func.count()).select_from(Predicao)) or 0
    odio = db.scalar(select(func.count()).select_from(Predicao).where(Predicao.label == 1)) or 0
    exemplos = db.scalar(select(func.count()).select_from(Exemplo)) or 0
    return Metricas(
        total_predicoes=total,
        taxa_odio=round(odio / total, 4) if total else 0.0,
        total_exemplos=exemplos,
        f1_modelo=None,
        ultimo_treino_em=None,
    )
```

Registrar em `app/main.py`: `app.include_router(metricas.router)`.

- [ ] **Step 6: Rodar os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 7: Commit**

```bash
git add app/ tests/
git commit -m "$(cat <<'EOF'
Adiciona GET /predicoes paginado e GET /metricas

Métricas são agregadas e públicas: total de predições, taxa de ódio e
total de exemplos ensinados.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Módulo de treino (`ml/`)

**Files:**
- Create: `ml/__init__.py`, `ml/dados.py`, `ml/treinar.py`, `tests/test_ml_dados.py`
- Delete: `dados_treino.py`, `retreinar_modelo.py`, `test_dados_treino.py`, `retreinar.bat`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `app.database.SessionLocal`, `app.models.Exemplo`
- Produces:
  - `ml.dados.carregar_dados(db=None, pasta_datasets=Path("HateBR-7.0.0/dataset")) -> pd.DataFrame` — colunas `comentario` e `label_final`
  - `ml.treinar.treinar(db=None) -> dict` — devolve `{"f1_macro", "desvio", "seed", "qtd_exemplos", "substituiu"}`

- [ ] **Step 1: Escrever os testes de `ml/dados.py`**

O `test_dados_treino.py` atual só roda se os datasets reais estiverem baixados. Estes
usam CSVs falsos, então rodam no CI.

`tests/test_ml_dados.py`:
```python
import pandas as pd
import pytest

from ml import dados


@pytest.fixture
def pasta_datasets(tmp_path):
    pd.DataFrame(
        {"comentario": ["te odeio", "bom dia"], "label_final": [1, 0]}
    ).to_csv(tmp_path / "HateBR.csv", index=False)
    pd.DataFrame(
        {
            "text": ["some morto", "que dia lindo"],
            "homophobia": [1, 0],
            "obscene": [1, 0],
            "insult": [0, 0],
            "racism": [0, 0],
            "misogyny": [0, 0],
            "xenophobia": [0, 0],
        }
    ).to_csv(tmp_path / "ToLD-BR.csv", index=False)
    return tmp_path


def test_carregar_dados_junta_os_dois_datasets(pasta_datasets):
    df = dados.carregar_dados(db=None, pasta_datasets=pasta_datasets)
    assert list(df.columns) == ["comentario", "label_final"]
    assert len(df) == 4
    assert set(df["label_final"].unique()) <= {0, 1}


def test_told_br_vira_odio_com_duas_categorias(pasta_datasets):
    """A regra de rotulagem do ToLD-BR: 2 ou mais categorias de toxicidade = ódio."""
    df = dados.carregar_dados(db=None, pasta_datasets=pasta_datasets)
    assert df[df["comentario"] == "some morto"]["label_final"].iloc[0] == 1
    assert df[df["comentario"] == "que dia lindo"]["label_final"].iloc[0] == 0


def test_carregar_dados_inclui_exemplos_do_banco(pasta_datasets, sessao, usuario):
    from app.models import Exemplo

    sessao.add(Exemplo(texto="frase ensinada", label=1, usuario_id=usuario.id))
    sessao.commit()
    df = dados.carregar_dados(db=sessao, pasta_datasets=pasta_datasets)
    assert len(df) == 5
    assert "frase ensinada" in df["comentario"].values


def test_dataset_faltando_levanta_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="HateBR.csv"):
        dados.carregar_dados(db=None, pasta_datasets=tmp_path)
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `pytest tests/test_ml_dados.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'ml'`

- [ ] **Step 3: Escrever o `ml/dados.py`**

```python
"""Monta o conjunto de treino: HateBR + ToLD-BR + exemplos ensinados no painel.

Sucessor do dados_treino.py. A diferença é a origem dos exemplos próprios:
antes um CSV, agora a tabela `exemplo`.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

CATEGORIAS_TOXICIDADE = [
    "homophobia",
    "obscene",
    "insult",
    "racism",
    "misogyny",
    "xenophobia",
]
PASTA_PADRAO = Path("HateBR-7.0.0/dataset")


def carregar_dados(db: Session | None = None, pasta_datasets: Path = PASTA_PADRAO) -> pd.DataFrame:
    pasta_datasets = Path(pasta_datasets)

    caminho_hatebr = pasta_datasets / "HateBR.csv"
    if not caminho_hatebr.exists():
        raise FileNotFoundError(
            f"HateBR.csv não encontrado em {pasta_datasets}. "
            "Baixe em https://github.com/franciellevargas/HateBR"
        )
    hatebr = pd.read_csv(caminho_hatebr)[["comentario", "label_final"]]

    caminho_told = pasta_datasets / "ToLD-BR.csv"
    if not caminho_told.exists():
        raise FileNotFoundError(
            f"ToLD-BR.csv não encontrado em {pasta_datasets}. "
            "Baixe em https://github.com/JAugusto97/ToLD-Br"
        )
    told = pd.read_csv(caminho_told)
    told["label_final"] = (told[CATEGORIAS_TOXICIDADE].sum(axis=1) >= 2).astype(int)
    told = told.rename(columns={"text": "comentario"})[["comentario", "label_final"]]

    partes = [hatebr, told]

    if db is not None:
        from app.models import Exemplo

        linhas = db.execute(select(Exemplo.texto, Exemplo.label)).all()
        if linhas:
            partes.append(
                pd.DataFrame(linhas, columns=["comentario", "label_final"])
            )

    return pd.concat(partes, ignore_index=True)
```

- [ ] **Step 4: Rodar os testes de dados**

Run: `pytest tests/test_ml_dados.py -v`
Expected: PASS, 4 testes.

- [ ] **Step 5: Escrever o `ml/treinar.py`**

Duas mudanças em relação ao `retreinar_modelo.py`: salva o pipeline inteiro em um
arquivo só, e devolve um dicionário em vez de imprimir, para que a Task 10 possa
gravar o resultado na tabela `treino`.

```python
"""Treina o classificador e substitui o artefato apenas se o F1 melhorar.

Sucessor do retreinar_modelo.py. Roda FORA do processo da API: precisa de
muita RAM por alguns minutos, o oposto do perfil de uma API.

Uso: python -m ml.treinar
"""

import random
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from ml.dados import PASTA_PADRAO, carregar_dados

CAMINHO_ARTEFATO = Path("ml/artefatos/modelo.pkl")


def construir_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def treinar(
    db: Session | None = None,
    pasta_datasets: Path = PASTA_PADRAO,
    caminho_artefato: Path = CAMINHO_ARTEFATO,
    f1_atual: float | None = None,
) -> dict:
    dados = carregar_dados(db=db, pasta_datasets=pasta_datasets)
    seed = random.randint(0, 999_999)
    pipeline = construir_pipeline()

    validador = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        dados["comentario"],
        dados["label_final"],
        cv=validador,
        scoring="f1_macro",
    )
    f1_macro = float(scores.mean())
    desvio = float(scores.std())

    substituiu = f1_atual is None or f1_macro > f1_atual
    if substituiu:
        pipeline.fit(dados["comentario"], dados["label_final"])
        caminho_artefato.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, caminho_artefato)

    return {
        "f1_macro": f1_macro,
        "desvio": desvio,
        "seed": seed,
        "qtd_exemplos": len(dados),
        "substituiu": substituiu,
    }


def main() -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        resultado = treinar(db=db)
    finally:
        db.close()
    print(f"F1 macro: {resultado['f1_macro']:.4f} (desvio: {resultado['desvio']:.4f})")
    print(f"Exemplos: {resultado['qtd_exemplos']}")
    print("Artefato substituído." if resultado["substituiu"] else "Artefato mantido.")


if __name__ == "__main__":
    main()
```

Nota sobre a direção do import: `ml/` importa de `app/` (para ler a tabela `exemplo`),
mas `app/` nunca importa de `ml/`. A seta aponta num sentido só.

- [ ] **Step 6: Treinar de verdade e conferir que a API passa a responder**

```bash
python -m ml.treinar
uvicorn app.main:app --reload
```
Abrir `http://127.0.0.1:8000/docs` e executar `POST /predicoes` com "eu te odeio".
Expected: `201` com `"rotulo": "Discurso de Ódio"` — não mais `503`.

- [ ] **Step 7: Versionar o artefato**

O `.gitignore` precisa deixar de ignorar o modelo, porque o deploy precisa dele na
imagem. Os datasets brutos continuam ignorados.

Remover do `.gitignore` as linhas que ignoram `*.pkl` e acrescentar:
```
# O artefato treinado é versionado de propósito: o deploy precisa dele.
# Os datasets originais continuam fora do repositório.
!ml/artefatos/modelo.pkl
```

- [ ] **Step 8: Remover os arquivos antigos**

```bash
git rm dados_treino.py retreinar_modelo.py test_dados_treino.py retreinar.bat
git rm --cached modelo_logistic_regression.pkl vetorizador.pkl best_score.json 2>/dev/null || true
rm -f modelo_logistic_regression.pkl vetorizador.pkl
```

- [ ] **Step 9: Rodar todos os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 10: Commit**

```bash
git add ml/ tests/ .gitignore
git commit -m "$(cat <<'EOF'
Move o treino para ml/, lendo os exemplos do banco

O treino salva o pipeline sklearn inteiro em um artefato só, em vez de
modelo e vetorizador separados, que podiam sair de sincronia. treinar()
devolve o resultado em vez de imprimir, para a rota de treino registrá-lo.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `POST /treinos` e `GET /treinos/{id}`

**Files:**
- Create: `app/routers/treinos.py`, `tests/test_treinos.py`
- Modify: `app/models.py`, `app/schemas.py`, `app/main.py`, `app/routers/metricas.py`

**Interfaces:**
- Consumes: `ml.treinar.treinar`, `app.security.usuario_atual`
- Produces:
  - `app.models.Treino` com `status` em `("pendente", "rodando", "concluido", "falhou")`
  - `TreinoSaida(id, status, f1_macro, desvio, qtd_exemplos, iniciado_em, terminado_em, erro)`

**Atenção à regra de import:** `app/routers/treinos.py` importa `ml.treinar` **dentro
da função** de background, não no topo do módulo. Isso mantém a API livre de pandas e
sklearn no caminho de import, que é o que permite o container de produção subir leve.

- [ ] **Step 1: Acrescentar a tabela `treino`**

Em `app/models.py`:
```python
from sqlalchemy import Enum as SAEnum

STATUS_TREINO = ("pendente", "rodando", "concluido", "falhou")


class Treino(Base):
    __tablename__ = "treino"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(
        SAEnum(*STATUS_TREINO, name="status_treino", native_enum=False),
        nullable=False,
        default="pendente",
    )
    f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    desvio: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qtd_exemplos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    erro: Mapped[str | None] = mapped_column(String(500), nullable=True)
    iniciado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    terminado_em: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
```

- [ ] **Step 2: Escrever os testes**

`tests/test_treinos.py`:
```python
import pytest

from app.models import Treino


@pytest.fixture
def treino_falso(monkeypatch):
    """Treinar de verdade leva minutos. O teste verifica o fluxo, não o sklearn."""
    from app.routers import treinos

    monkeypatch.setattr(
        treinos,
        "_executar_treino",
        lambda treino_id, db_factory: None,
    )


def test_criar_treino_exige_token(cliente):
    assert cliente.post("/treinos").status_code == 401


def test_criar_treino_devolve_202_e_id(cliente_logado, treino_falso):
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 202
    assert "id" in resposta.json()


def test_criar_treino_com_um_ja_rodando_devolve_409(cliente_logado, sessao, treino_falso):
    sessao.add(Treino(status="rodando"))
    sessao.commit()
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 409


def test_consultar_treino_inexistente_devolve_404(cliente_logado):
    assert cliente_logado.get("/treinos/999").status_code == 404


def test_consultar_treino_devolve_status(cliente_logado, sessao):
    treino = Treino(status="concluido", f1_macro=0.75, desvio=0.01, qtd_exemplos=100)
    sessao.add(treino)
    sessao.commit()
    resposta = cliente_logado.get(f"/treinos/{treino.id}")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "concluido"
    assert resposta.json()["f1_macro"] == 0.75


def test_treino_desligado_devolve_503(cliente_logado, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "TREINO_HABILITADO", False)
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 503
    assert "produção" in resposta.json()["detail"].lower()
```

- [ ] **Step 3: Rodar e confirmar que falham**

Run: `pytest tests/test_treinos.py -v`
Expected: FAIL, `404` — as rotas não existem.

- [ ] **Step 4: Acrescentar os schemas**

Em `app/schemas.py`:
```python
class TreinoCriado(BaseModel):
    id: int
    status: str


class TreinoSaida(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    f1_macro: float | None
    desvio: float | None
    qtd_exemplos: int | None
    erro: str | None
    iniciado_em: datetime
    terminado_em: datetime | None
```

- [ ] **Step 5: Escrever o router**

`app/routers/treinos.py`:
```python
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, get_db
from app.models import Treino, Usuario
from app.schemas import TreinoCriado, TreinoSaida
from app.security import usuario_atual

router = APIRouter(prefix="/treinos", tags=["treinos"])


def _executar_treino(treino_id: int, db_factory=SessionLocal) -> None:
    """Roda em background, com sessão própria.

    A sessão do request já foi fechada quando isto executa, por isso abre outra.
    O import de ml.treinar fica aqui dentro de propósito: assim pandas e sklearn
    não entram no caminho de import da API.
    """
    from ml.treinar import treinar

    db = db_factory()
    try:
        treino = db.get(Treino, treino_id)
        treino.status = "rodando"
        db.commit()

        melhor = db.scalar(
            select(Treino.f1_macro)
            .where(Treino.status == "concluido")
            .order_by(Treino.f1_macro.desc())
            .limit(1)
        )
        resultado = treinar(db=db, f1_atual=melhor)

        treino.status = "concluido"
        treino.f1_macro = resultado["f1_macro"]
        treino.desvio = resultado["desvio"]
        treino.seed = resultado["seed"]
        treino.qtd_exemplos = resultado["qtd_exemplos"]
        treino.terminado_em = datetime.now(UTC)
        db.commit()
    except Exception as exc:  # noqa: BLE001 - o erro precisa virar registro, não sumir
        treino = db.get(Treino, treino_id)
        treino.status = "falhou"
        treino.erro = str(exc)[:500]
        treino.terminado_em = datetime.now(UTC)
        db.commit()
    finally:
        db.close()


@router.post("", response_model=TreinoCriado, status_code=status.HTTP_202_ACCEPTED)
def criar_treino(
    tarefas: BackgroundTasks,
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> TreinoCriado:
    if not settings.TREINO_HABILITADO:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retreino desligado em produção: o plano gratuito tem 512 MB de RAM, "
                "insuficiente para validação cruzada. Rode localmente com "
                "`python -m ml.treinar`."
            ),
        )

    rodando = db.scalar(select(Treino).where(Treino.status.in_(("pendente", "rodando"))))
    if rodando is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Já existe um treino em andamento (id {rodando.id})",
        )

    treino = Treino(status="pendente")
    db.add(treino)
    db.commit()
    tarefas.add_task(_executar_treino, treino.id, SessionLocal)
    return TreinoCriado(id=treino.id, status=treino.status)


@router.get("/{treino_id}", response_model=TreinoSaida)
def obter_treino(
    treino_id: int,
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> Treino:
    treino = db.get(Treino, treino_id)
    if treino is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Treino não encontrado")
    return treino
```

Registrar em `app/main.py`: `app.include_router(treinos.router)`.

- [ ] **Step 6: Ligar o F1 nas métricas**

Em `app/routers/metricas.py`, substituir os dois `None`:
```python
from app.models import Treino

ultimo = db.scalar(
    select(Treino)
    .where(Treino.status == "concluido")
    .order_by(Treino.terminado_em.desc())
    .limit(1)
)
...
    f1_modelo=ultimo.f1_macro if ultimo else None,
    ultimo_treino_em=ultimo.terminado_em if ultimo else None,
```

- [ ] **Step 7: Rodar os testes**

Run: `pytest -v`
Expected: PASS, todos.

- [ ] **Step 8: Gerar a migration**

```bash
alembic revision --autogenerate -m "cria tabela treino"
alembic upgrade head
```

- [ ] **Step 9: Commit**

```bash
git add app/ alembic/versions/ tests/
git commit -m "$(cat <<'EOF'
Adiciona POST /treinos em background e GET /treinos/{id}

A tabela treino substitui o best_score.json e o historico_execucoes.log.
O 202 é honesto porque existe onde consultar o desfecho. Em produção a
rota devolve 503, porque 512 MB não comportam validação cruzada.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Containers

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `entrypoint.sh`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: todo o código anterior
- Produces: `docker compose up` sobe Postgres e API, com migrations aplicadas no start

- [ ] **Step 1: Escrever o `.dockerignore`**

```
.venv/
.git/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
HateBR-7.0.0/
docs/
tests/
.env
```

Nota: `HateBR-7.0.0/` fica de fora porque a imagem só serve predições. Quem treina
é você, localmente.

- [ ] **Step 2: Escrever o `Dockerfile`**

Multi-stage: as dependências são instaladas em uma camada separada, então mudar o
código não reinstala tudo.

```dockerfile
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ml/ ./ml/
COPY alembic/ ./alembic/
COPY alembic.ini entrypoint.sh ./

RUN chmod +x entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
```

Nota de segurança: o container roda como usuário `app`, não como root. É o tipo de
detalhe que aparece em revisão de código de verdade.

- [ ] **Step 3: Escrever o `entrypoint.sh`**

```bash
#!/bin/sh
set -e

echo "Aplicando migrations..."
alembic upgrade head

echo "Subindo a API..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

- [ ] **Step 4: Completar o `docker-compose.yml`**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: hatebr
      POSTGRES_PASSWORD: hatebr
      POSTGRES_DB: hatebr
    ports:
      - "5432:5432"
    volumes:
      - dados_postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hatebr"]
      interval: 5s
      retries: 5

  api:
    build: .
    environment:
      DATABASE_URL: postgresql+psycopg://hatebr:hatebr@db:5432/hatebr
      JWT_SECRET: segredo-de-desenvolvimento-nao-usar-em-producao
      TREINO_HABILITADO: "true"
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy

volumes:
  dados_postgres:
```

- [ ] **Step 5: Subir e testar**

```bash
docker compose up --build
```
Expected: log mostra as migrations aplicadas e o uvicorn subindo. Abrir
`http://localhost:8000/docs` e executar `POST /predicoes`.

- [ ] **Step 6: Conferir que o `/health` responde**

Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok","modelo":true}`

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore entrypoint.sh docker-compose.yml
git commit -m "$(cat <<'EOF'
Empacota a API em container com Postgres no compose

O entrypoint aplica as migrations antes de subir o uvicorn. O container
roda como usuário sem privilégio, não como root.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Integração contínua

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/dependabot.yml`

**Interfaces:**
- Consumes: `requirements-dev.txt`, `pytest`, `ruff`
- Produces: check verde obrigatório em cada push e PR

- [ ] **Step 1: Escrever o workflow**

O CI roda contra Postgres real, e não SQLite, justamente para pegar diferença de
dialeto antes de chegar em produção.

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  testes:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: hatebr
          POSTGRES_PASSWORD: hatebr
          POSTGRES_DB: hatebr_teste
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 5s
          --health-timeout 5s
          --health-retries 5

    env:
      DATABASE_URL: postgresql+psycopg://hatebr:hatebr@localhost:5432/hatebr_teste
      JWT_SECRET: segredo-de-teste
      RATE_LIMIT_ATIVO: "false"

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Instalar dependências
        run: pip install -r requirements-dev.txt

      - name: Lint
        run: ruff check .

      - name: Formatação
        run: ruff format --check .

      - name: Migrations aplicam sem erro
        run: alembic upgrade head

      - name: Testes
        run: pytest --cov=app --cov-report=term-missing

  imagem:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: A imagem Docker builda
        run: docker build -t hatebr-api .
```

- [ ] **Step 2: Escrever o Dependabot**

`.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
    open-pull-requests-limit: 5

  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
```

- [ ] **Step 3: Commit e push**

```bash
git add .github/
git commit -m "$(cat <<'EOF'
Adiciona CI no GitHub Actions e Dependabot

Os testes rodam contra Postgres real no CI, não SQLite, para diferença de
dialeto aparecer no PR e não em produção.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 4: Conferir que o CI passou**

Abrir a aba Actions em `https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio/actions`.
Expected: os dois jobs verdes. Se falhar, corrigir antes de seguir — a Task 13 depende
de um CI confiável.

---

### Task 13: Deploy no Render

**Files:**
- Create: `render.yaml`

**Interfaces:**
- Consumes: `Dockerfile`, `/health`
- Produces: URL pública com o Swagger aberto

**Esta task exige ação sua:** criar conta no Render e autorizar o acesso ao
repositório. A conexão OAuth não pode ser automatizada.

- [ ] **Step 1: Escrever o `render.yaml`**

```yaml
services:
  - type: web
    name: hatebr-api
    runtime: docker
    plan: free
    healthCheckPath: /health
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: hatebr-db
          property: connectionString
      - key: JWT_SECRET
        generateValue: true
      - key: TREINO_HABILITADO
        value: "false"
      - key: RATE_LIMIT_ATIVO
        value: "true"

databases:
  - name: hatebr-db
    plan: free
```

Nota: `generateValue: true` faz o Render criar o `JWT_SECRET` sozinho. O segredo não
passa por você, nem por este repositório, nem por nenhuma conversa.

Nota: `TREINO_HABILITADO: "false"` é a decisão registrada na spec — 512 MB não
comportam validação cruzada sobre 28 mil linhas.

- [ ] **Step 2: Ajustar o `DATABASE_URL` do Render**

O Render entrega a connection string no formato `postgres://`, que o SQLAlchemy 2.0
não aceita. Acrescentar a normalização em `app/config.py`:

```python
from pydantic import field_validator


class Settings(BaseSettings):
    ...

    @field_validator("DATABASE_URL")
    @classmethod
    def normalizar_url(cls, v: str) -> str:
        """O Render entrega postgres://; o SQLAlchemy 2.0 exige o driver explícito."""
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg://", 1)
        return v
```

Escrever o teste correspondente em `tests/test_config.py`:
```python
from app.config import Settings


def test_url_do_render_e_normalizada():
    s = Settings(DATABASE_URL="postgres://u:p@host:5432/db")
    assert s.DATABASE_URL == "postgresql+psycopg://u:p@host:5432/db"


def test_url_ja_correta_nao_e_alterada():
    url = "postgresql+psycopg://u:p@host:5432/db"
    assert Settings(DATABASE_URL=url).DATABASE_URL == url
```

Run: `pytest tests/test_config.py -v`
Expected: PASS, 2 testes.

- [ ] **Step 3: Commit e push**

```bash
git add render.yaml app/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Adiciona configuração de deploy no Render

Normaliza a connection string postgres:// que o Render entrega para o
formato com driver explícito que o SQLAlchemy 2.0 exige.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 4: Criar o serviço no Render (ação sua)**

1. Criar conta em `https://render.com`.
2. "New" → "Blueprint" → conectar o repositório `ZazzySaint64/IA-Para-Discurso-de-dio`.
3. O Render lê o `render.yaml` e cria o banco e o serviço web.
4. Aguardar o primeiro deploy (alguns minutos: a imagem precisa buildar).

- [ ] **Step 5: Criar o usuário no banco de produção**

Pegar a "External Database URL" no painel do Render e rodar localmente, uma vez:

```bash
DATABASE_URL="<external database url do Render>" python -m app.seed --gerar-senha --email <seu email>
```
Expected: imprime a senha uma vez. **Salvar no gerenciador de senhas.**

- [ ] **Step 6: Conferir que está no ar**

```bash
curl https://hatebr-api.onrender.com/health
```
Expected: `{"status":"ok","modelo":true}`

Abrir `https://hatebr-api.onrender.com/docs`, clicar em **Authorize**, entrar com o
email e a senha, e executar `GET /exemplos`.
Expected: `200`.

Nota: o plano gratuito hiberna depois de 15 minutos sem tráfego. A primeira
requisição depois disso leva cerca de 50 segundos. Isso vai documentado no README,
para ninguém achar que está quebrado.

---

### Task 14: Streamlit como cliente HTTP

**Files:**
- Create: `frontend/app.py`, `frontend/requirements.txt`
- Delete: `classificador.py`, `painel_treino.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: a API por HTTP, nunca por import
- Produces: interface com a tela pública e o painel de treino

**Regra desta task:** `frontend/app.py` não importa `app` nem `ml`, e não carrega
`.pkl`. Se importar, a separação de camadas que o projeto inteiro demonstra deixa de
existir. Só `requests`.

- [ ] **Step 1: Escrever o `frontend/requirements.txt`**

```
streamlit>=1.39
requests>=2.32
```

- [ ] **Step 2: Escrever o `frontend/app.py`**

```python
"""Cliente Streamlit da API de detecção de discurso de ódio.

Este arquivo NÃO carrega o modelo e NÃO acessa o banco. Ele só fala HTTP
com a API. Foi exatamente essa separação que motivou o redesenho do projeto.
"""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60  # o plano gratuito do Render hiberna; a primeira chamada demora

st.set_page_config(page_title="Detector de Discurso de Ódio", page_icon="🛡️")


def pedir(metodo: str, rota: str, **kwargs) -> requests.Response | None:
    try:
        return requests.request(metodo, f"{API}{rota}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Não consegui falar com a API: {exc}")
        return None


aba_publica, aba_treino = st.tabs(["Classificar", "Painel de treino"])

with aba_publica:
    st.title("Detector de Discurso de Ódio")
    texto = st.text_area("Comentário", max_chars=1000)
    if st.button("Classificar", disabled=not texto.strip()):
        resposta = pedir("POST", "/predicoes", json={"texto": texto})
        if resposta is None:
            pass
        elif resposta.status_code == 201:
            dado = resposta.json()
            if dado["label"] == 1:
                st.error(f"{dado['rotulo']} — confiança {dado['confianca']:.0%}")
            else:
                st.success(f"{dado['rotulo']} — confiança {dado['confianca']:.0%}")
        else:
            st.warning(resposta.json().get("detail", "Erro inesperado"))

    metricas = pedir("GET", "/metricas")
    if metricas is not None and metricas.status_code == 200:
        m = metricas.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Classificações feitas", m["total_predicoes"])
        c2.metric("Taxa de ódio", f"{m['taxa_odio']:.0%}")
        c3.metric("F1 do modelo", f"{m['f1_modelo']:.3f}" if m["f1_modelo"] else "—")

with aba_treino:
    st.title("Painel de treino")

    if "token" not in st.session_state:
        st.session_state.token = None

    if st.session_state.token is None:
        with st.form("login"):
            email = st.text_input("Email")
            senha = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar"):
                resposta = pedir(
                    "POST", "/auth/login", data={"username": email, "password": senha}
                )
                if resposta is not None and resposta.status_code == 200:
                    st.session_state.token = resposta.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Email ou senha incorretos.")
    else:
        cabecalho = {"Authorization": f"Bearer {st.session_state.token}"}

        with st.form("ensinar"):
            novo = st.text_area("Comentário para ensinar", max_chars=1000)
            rotulo = st.radio("É discurso de ódio?", ["Não", "Sim"], horizontal=True)
            if st.form_submit_button("Guardar este exemplo"):
                resposta = pedir(
                    "POST",
                    "/exemplos",
                    json={"texto": novo, "label": 1 if rotulo == "Sim" else 0},
                    headers=cabecalho,
                )
                if resposta is None:
                    pass
                elif resposta.status_code == 201:
                    st.success("Exemplo guardado.")
                elif resposta.status_code == 409:
                    st.warning("Esse texto já foi ensinado antes.")
                elif resposta.status_code == 401:
                    st.session_state.token = None
                    st.warning("Sessão expirada, entre de novo.")
                    st.rerun()
                else:
                    st.error(resposta.json().get("detail", "Erro inesperado"))

        if st.button("Atualizar modelo agora"):
            resposta = pedir("POST", "/treinos", headers=cabecalho)
            if resposta is None:
                pass
            elif resposta.status_code == 202:
                st.info(f"Treino iniciado (id {resposta.json()['id']}).")
            else:
                st.warning(resposta.json().get("detail", "Erro inesperado"))

        exemplos = pedir("GET", "/exemplos?limite=20", headers=cabecalho)
        if exemplos is not None and exemplos.status_code == 200:
            st.subheader(f"Exemplos ensinados ({exemplos.json()['total']})")
            st.dataframe(exemplos.json()["itens"], use_container_width=True)

        if st.button("Sair"):
            st.session_state.token = None
            st.rerun()
```

- [ ] **Step 3: Acrescentar o serviço no compose**

```yaml
  frontend:
    image: python:3.12-slim
    working_dir: /front
    volumes:
      - ./frontend:/front
    command: >
      sh -c "pip install --no-cache-dir -r requirements.txt &&
             streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
    environment:
      API_URL: http://api:8000
    ports:
      - "8501:8501"
    depends_on:
      - api
```

- [ ] **Step 4: Testar de ponta a ponta**

```bash
docker compose up --build
```
Abrir `http://localhost:8501`. Classificar um comentário na primeira aba; entrar com
o usuário do seed na segunda; ensinar um exemplo; tentar ensinar o mesmo de novo.
Expected: classificação funciona, login funciona, exemplo é guardado, repetido avisa
"já foi ensinado antes".

- [ ] **Step 5: Remover os Streamlit antigos**

```bash
git rm classificador.py painel_treino.py
```

- [ ] **Step 6: Commit**

```bash
git add frontend/ docker-compose.yml
git commit -m "$(cat <<'EOF'
Reescreve o Streamlit como cliente HTTP da API

O frontend não carrega mais o .pkl nem lê arquivo: fala só HTTP. As duas
telas viraram abas, e o painel de treino autentica por JWT.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: README

O README é a parte do projeto que o recrutador lê primeiro, e às vezes a única.

**Files:**
- Modify: `README.md`
- Create: `docs/arquitetura.svg`

**Interfaces:**
- Consumes: tudo
- Produces: documentação de entrada do repositório

- [ ] **Step 1: Escrever a estrutura do README**

Seções, nesta ordem:

1. **Título, badges** (Python, FastAPI, Postgres, Docker, licença, status do CI) e uma
   linha em inglês, como já existe hoje.
2. **Link para a API no ar**, com aviso sobre a hibernação de 50 segundos do plano
   gratuito, e link direto para `/docs`.
3. **O que é**, dois parágrafos.
4. **Arquitetura**, com o diagrama e a regra `app/` nunca importa `ml/` explicada em
   uma frase.
5. **Tabela de endpoints** — copiar do contrato da spec.
6. **Como rodar**: `docker compose up`, `python -m app.seed --gerar-senha`,
   `python -m ml.treinar`.
7. **Decisões de projeto**, a seção que mais pesa numa entrevista. Uma linha por
   decisão, com o porquê:
   - SHA-256 em env var virou bcrypt no banco (rapidez é defeito em hash de senha).
   - CSV, JSON e log soltos viraram tabelas (`UNIQUE` no texto impede exemplo repetido).
   - Modelo e vetorizador separados viraram um pipeline só (não saem de sincronia).
   - Treino separado da API (perfis de recurso opostos; e cabe em 512 MB).
   - Retreino desligado em produção, com o motivo, e não escondido.
   - Predição não guarda IP nem identificação de quem digitou (LGPD).
8. **O que ficou de fora e por quê** — copiar de "Fora de escopo" da spec.
9. **Licenças dos datasets**, HateBR e ToLD-BR, com links, como já está hoje.

- [ ] **Step 2: Desenhar o diagrama**

Substituir `docs/fluxo.svg` por `docs/arquitetura.svg`, mostrando: Streamlit → HTTP →
FastAPI → Postgres, com `ml/treinar.py` escrevendo o `modelo.pkl` de fora e uma seta
pontilhada indicando que a API só lê o artefato.

- [ ] **Step 3: Atualizar as capturas de tela**

As três imagens em `docs/` são das telas antigas. Tirar novas: o Swagger em `/docs`
com o botão Authorize, e as duas abas do Streamlit novo.

- [ ] **Step 4: Conferir que as instruções funcionam em uma máquina limpa**

Seguir o próprio README do zero, em uma pasta nova:
```bash
git clone https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio.git teste-limpo
cd teste-limpo
docker compose up --build
```
Expected: sobe sem nenhum passo que não esteja escrito no README. Qualquer passo que
você precisou adivinhar é um defeito do README — corrija.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git commit -m "$(cat <<'EOF'
Reescreve o README para o projeto como serviço

Inclui link da API no ar, tabela de endpoints, diagrama de arquitetura e
a seção de decisões de projeto com o porquê de cada uma.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Verificação final

- [ ] `pytest --cov=app` passa, com cobertura acima de 80% em `app/`
- [ ] `ruff check .` e `ruff format --check .` limpos
- [ ] `docker compose up --build` sobe API, banco e frontend do zero
- [ ] CI verde na `main`
- [ ] `https://<app>.onrender.com/health` devolve `200`
- [ ] `https://<app>.onrender.com/docs` abre, o Authorize funciona e uma rota protegida responde
- [ ] `POST /treinos` em produção devolve `503` com a mensagem explicativa
- [ ] Nenhum arquivo em `app/` importa de `ml/` — conferir com:
      `grep -rn "^from ml\|^import ml" app/` (deve não retornar nada; o único import de
      `ml` está dentro da função `_executar_treino`)
- [ ] `README.md` não menciona `classificador.py`, `painel_treino.py`, `gerar_hash_senha.py`,
      `retreinar_modelo.py`, `dados_treino.py`, `best_score.json` ou `HATEBR_SENHA_HASH`
