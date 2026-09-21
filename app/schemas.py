from pydantic import BaseModel, Field, field_validator


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
