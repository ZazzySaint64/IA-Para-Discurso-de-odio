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
    if len(senha.encode("utf-8")) > LIMITE_BCRYPT_BYTES:
        raise ValueError(f"senha não pode passar de {LIMITE_BCRYPT_BYTES} bytes")
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


def usuario_atual(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> Usuario:
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
