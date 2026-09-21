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
