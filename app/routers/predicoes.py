from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import ml
from app.database import get_db
from app.limites import limiter
from app.models import Predicao, Usuario
from app.schemas import Pagina, PredicaoEntrada, PredicaoRegistro, PredicaoSaida
from app.security import usuario_atual

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
