import pytest
from pydantic import ValidationError

from app.config import Settings


def test_jwt_secret_curto_e_rejeitado():
    with pytest.raises(ValidationError):
        Settings(JWT_SECRET="curto-demais")


def test_jwt_secret_com_32_bytes_ou_mais_e_aceito():
    segredo = "x" * 32
    assert Settings(JWT_SECRET=segredo).JWT_SECRET == segredo


def test_url_do_render_e_normalizada():
    s = Settings(DATABASE_URL="postgres://u:p@host:5432/db")
    assert s.DATABASE_URL == "postgresql+psycopg://u:p@host:5432/db"


def test_url_ja_correta_nao_e_alterada():
    url = "postgresql+psycopg://u:p@host:5432/db"
    assert Settings(DATABASE_URL=url).DATABASE_URL == url
