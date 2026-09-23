from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

STATUS_TREINO = ("pendente", "rodando", "concluido", "falhou")


class Predicao(Base):
    __tablename__ = "predicao"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    texto: Mapped[str] = mapped_column(String(1000), nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    confianca: Mapped[float] = mapped_column(Float, nullable=False)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Exemplo(Base):
    __tablename__ = "exemplo"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    texto: Mapped[str] = mapped_column(String(1000), unique=True, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuario.id"), nullable=False)
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


class Treino(Base):
    __tablename__ = "treino"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(
        SAEnum(*STATUS_TREINO, name="status_treino", native_enum=False, create_constraint=True),
        nullable=False,
        default="pendente",
    )
    f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
    desvio: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qtd_exemplos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Só "concluido" preenche isto. "falhou" também fica None mesmo quando o
    # treino chegou a rodar: é o mesmo caminho que já perde o f1_macro quando
    # a recarga do .pkl falha (ver "Limitações conhecidas" no README).
    substituiu: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    erro: Mapped[str | None] = mapped_column(String(500), nullable=True)
    iniciado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    terminado_em: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
