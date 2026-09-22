import pytest
from pydantic import ValidationError

from app.config import Settings


def test_jwt_secret_curto_e_rejeitado():
    with pytest.raises(ValidationError):
        Settings(JWT_SECRET="curto-demais")


def test_jwt_secret_com_32_bytes_ou_mais_e_aceito():
    segredo = "x" * 32
    assert Settings(JWT_SECRET=segredo).JWT_SECRET == segredo
