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
