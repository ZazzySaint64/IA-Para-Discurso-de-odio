from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def sem_espaco_sobrando(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("texto não pode ser só espaço em branco")
    return v


# Mesma regra em /predicoes e /exemplos: um alias em vez de duas cópias.
Texto = Annotated[str, Field(min_length=1, max_length=1000), AfterValidator(sem_espaco_sobrando)]


class PredicaoEntrada(BaseModel):
    texto: Texto


class PredicaoSaida(BaseModel):
    label: int
    rotulo: str
    confianca: float


class Token(BaseModel):
    access_token: str
    token_type: str


class ExemploEntrada(BaseModel):
    texto: Texto
    label: Literal[0, 1]


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


class TreinoCriado(BaseModel):
    id: int
    status: str


class TreinoSaida(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    f1_macro: float | None
    desvio: float | None
    qtd_exemplos: int | None
    erro: str | None
    iniciado_em: datetime
    terminado_em: datetime | None
