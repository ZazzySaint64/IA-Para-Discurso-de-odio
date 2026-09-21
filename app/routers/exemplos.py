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
