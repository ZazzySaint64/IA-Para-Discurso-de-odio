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


class Usuario(Base):
    __tablename__ = "usuario"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    senha_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
