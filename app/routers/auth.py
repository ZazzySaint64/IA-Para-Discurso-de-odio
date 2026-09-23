from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import security
from app.database import get_db
from app.models import Usuario
from app.schemas import Token

router = APIRouter(prefix="/auth", tags=["auth"])

# Hash descartável, com o mesmo custo de um hash real. Serve para que o login
# gaste o mesmo tempo existindo ou não o email, fechando a enumeração por tempo.
HASH_FALSO = security.gerar_hash("hash-descartavel-para-tempo-constante")


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    usuario = db.scalar(select(Usuario).where(Usuario.email == form.username))
    hash_para_conferir = usuario.senha_hash if usuario is not None else HASH_FALSO
    senha_confere = security.conferir_senha(form.password, hash_para_conferir)
    if usuario is None or not senha_confere:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=security.criar_token(usuario.id), token_type="bearer")
