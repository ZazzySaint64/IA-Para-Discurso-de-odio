from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://hatebr:hatebr@localhost:5432/hatebr"
    JWT_SECRET: str = "troque-isso-em-producao"
    JWT_EXPIRA_MINUTOS: int = 60
    MODELO_PATH: str = "ml/artefatos/modelo.pkl"
    TREINO_HABILITADO: bool = True
    RATE_LIMIT_ATIVO: bool = True


settings = Settings()
