import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import ml
from app.config import settings
from app.database import get_db
from app.limites import limiter
from app.routers import auth, exemplos, metricas, predicoes, treinos

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O modelo é carregado uma vez por processo, não a cada request.
    caminho = Path(settings.MODELO_PATH)
    if caminho.exists():
        ml.carregar_modelo(caminho)
        logger.info("Modelo carregado de %s", caminho)
    else:
        # Sem log, isto aparece só como health check falhando, sem dizer por quê.
        logger.error("Modelo não encontrado em %s: /health vai responder 503", caminho)
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
    # loc == ("body",) quando o corpo inteiro tem o tipo errado: sem nome de
    # campo, o prefixo viraria um ":" solto na frente da mensagem.
    detalhe = f"{campo}: {primeiro['msg']}" if campo else primeiro["msg"]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": detalhe},
    )


@app.exception_handler(RateLimitExceeded)
async def erro_de_limite(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Muitas requisições. Tente de novo em instantes."},
        headers={"Retry-After": str(exc.limit.limit.get_expiry())},
    )


@app.exception_handler(Exception)
async def erro_inesperado(request: Request, exc: Exception):
    """Garante que todo erro tenha o mesmo formato, inclusive os não previstos."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno no servidor"},
    )


@app.get("/", include_in_schema=False)
def raiz():
    """Uma API REST não tem homepage: manda quem chegar aqui pra doc interativa."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["infra"])
def health(db: Session = Depends(get_db)):
    if not ml.modelo_carregado():
        return JSONResponse(status_code=503, content={"detail": "Modelo não carregado"})
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(status_code=503, content={"detail": "Banco de dados indisponível"})
    return {"status": "ok", "modelo": True, "banco": True}


app.include_router(auth.router)
app.include_router(exemplos.router)
app.include_router(metricas.router)
app.include_router(predicoes.router)
app.include_router(treinos.router)
