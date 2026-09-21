from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app import ml
from app.config import settings
from app.limites import limiter
from app.routers import auth, exemplos, metricas, predicoes, treinos


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O modelo é carregado uma vez por processo, não a cada request.
    caminho = Path(settings.MODELO_PATH)
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


@app.exception_handler(Exception)
async def erro_inesperado(request: Request, exc: Exception):
    """Garante que todo erro tenha o mesmo formato, inclusive os não previstos."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno no servidor"},
    )


@app.get("/health", tags=["infra"])
def health():
    if not ml.modelo_carregado():
        return JSONResponse(status_code=503, content={"detail": "Modelo não carregado"})
    return {"status": "ok", "modelo": True}


app.include_router(auth.router)
app.include_router(exemplos.router)
app.include_router(metricas.router)
app.include_router(predicoes.router)
app.include_router(treinos.router)
