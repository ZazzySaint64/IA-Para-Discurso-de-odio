from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MINIMO_BYTES_JWT_SECRET = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://hatebr:hatebr@localhost:5432/hatebr"
    # Sem default de propósito: este repositório é público, e um default
    # conhecido passaria pelo validador de tamanho abaixo sem proteger nada.
    JWT_SECRET: str
    JWT_EXPIRA_MINUTOS: int = 60
    MODELO_PATH: str = "ml/artefatos/modelo.pkl"
    TREINO_HABILITADO: bool = True
    RATE_LIMIT_ATIVO: bool = True

    @field_validator("JWT_SECRET")
    @classmethod
    def segredo_forte_o_bastante(cls, v: str) -> str:
        """HS256 é HMAC-SHA256: chave menor que 32 bytes enfraquece a assinatura."""
        if len(v.encode("utf-8")) < MINIMO_BYTES_JWT_SECRET:
            raise ValueError(
                f"JWT_SECRET precisa de pelo menos {MINIMO_BYTES_JWT_SECRET} bytes. "
                'Gere um com: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def normalizar_url(cls, v: str) -> str:
        """O Render entrega a URL do Postgres como postgres:// em alguns lugares
        e postgresql:// (com "ql") em outros — os dois carecem do driver
        explícito que o SQLAlchemy 2.0 exige, senão ele cai no psycopg2, que
        este projeto não instala (instala psycopg[binary], o psycopg 3)."""
        if v.startswith("postgresql+"):
            return v
        if v.startswith("postgres://"):
            return "postgresql+psycopg://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            return "postgresql+psycopg://" + v[len("postgresql://") :]
        return v


settings = Settings()
