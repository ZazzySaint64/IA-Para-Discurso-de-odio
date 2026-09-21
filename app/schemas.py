from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredicaoEntrada(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)

    @field_validator("texto")
    @classmethod
    def sem_espaco_sobrando(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("texto não pode ser só espaço em branco")
        return v


class PredicaoSaida(BaseModel):
    label: int
    rotulo: str
    confianca: float


class Token(BaseModel):
    access_token: str
    token_type: str


class ExemploEntrada(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)
    label: Literal[0, 1]

    @field_validator("texto")
    @classmethod
    def sem_espaco_sobrando(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("texto não pode ser só espaço em branco")
        return v


class ExemploSaida(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    texto: str
    label: int
    criado_em: datetime


class PredicaoRegistro(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    texto: str
    label: int
    confianca: float
    criado_em: datetime


class Metricas(BaseModel):
    total_predicoes: int
    taxa_odio: float
    total_exemplos: int
    f1_modelo: float | None
    ultimo_treino_em: datetime | None


class Pagina[T](BaseModel):
    total: int
    itens: list[T]
