from app import security


def test_hash_nao_guarda_a_senha_em_texto():
    hash_ = security.gerar_hash("minha-senha-secreta")
    assert "minha-senha-secreta" not in hash_
    assert hash_.startswith("$2b$")


def test_hashes_da_mesma_senha_sao_diferentes():
    """bcrypt embute salt: duas chamadas com a mesma senha dão hashes distintos."""
    assert security.gerar_hash("abc123") != security.gerar_hash("abc123")


def test_conferir_senha_certa_e_errada():
    hash_ = security.gerar_hash("abc123")
    assert security.conferir_senha("abc123", hash_) is True
    assert security.conferir_senha("abc124", hash_) is False


def test_token_carrega_o_id_do_usuario():
    token = security.criar_token(42)
    assert security.ler_token(token) == 42


def test_token_invalido_devolve_none():
    assert security.ler_token("token.falsificado.aqui") is None


def test_login_com_credencial_certa(cliente, usuario):
    resposta = cliente.post(
        "/auth/login", data={"username": usuario.email, "password": "senha-de-teste"}
    )
    assert resposta.status_code == 200
    assert "access_token" in resposta.json()


def test_login_com_senha_errada(cliente, usuario):
    resposta = cliente.post("/auth/login", data={"username": usuario.email, "password": "errada"})
    assert resposta.status_code == 401


def test_login_com_email_inexistente(cliente):
    resposta = cliente.post(
        "/auth/login", data={"username": "ninguem@exemplo.com", "password": "x"}
    )
    assert resposta.status_code == 401


def test_gerar_hash_rejeita_senha_maior_que_72_bytes():
    """bcrypt trunca silenciosamente acima de 72 bytes; o guard evita isso."""
    senha_73_bytes = "a" * 73
    try:
        security.gerar_hash(senha_73_bytes)
    except ValueError:
        pass
    else:
        raise AssertionError("esperava ValueError para senha acima de 72 bytes")
