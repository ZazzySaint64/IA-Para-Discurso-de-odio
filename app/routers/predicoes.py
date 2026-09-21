from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app import ml
from app.database import get_db
from app.limites import limiter
from app.models import Predicao
from app.schemas import PredicaoEntrada, PredicaoSaida

router = APIRouter(tags=["predicoes"])

ROTULOS = {0: "Não é Discurso de Ódio", 1: "Discurso de Ódio"}


@router.post("/predicoes", response_model=PredicaoSaida, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
def criar_predicao(
    request: Request, entrada: PredicaoEntrada, db: Session = Depends(get_db)
) -> PredicaoSaida:
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
