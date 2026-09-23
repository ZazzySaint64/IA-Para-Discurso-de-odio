import pytest
from sqlalchemy import select

from app import security, seed
from app.models import Exemplo, Usuario


def test_senha_gerada_tem_entropia_suficiente():
    senha = seed.gerar_senha()
    assert len(senha) >= 20
    assert seed.gerar_senha() != seed.gerar_senha()


def test_criar_usuario_guarda_hash_nao_a_senha(sessao):
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "minha-senha")
    assert usuario.senha_hash != "minha-senha"
    assert sessao.get(Usuario, usuario.id) is not None


def test_criar_usuario_repetido_levanta_erro(sessao):
    seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    with pytest.raises(seed.UsuarioJaExisteError):
        seed.criar_usuario(sessao, "eu@exemplo.com", "abc")


def test_importar_csv_traz_os_exemplos(sessao, tmp_path):
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text("comentario,label_final\nte odeio,1\nbom dia,0\n", encoding="utf-8")
    quantos = seed.importar_csv(sessao, csv, usuario.id)
    assert quantos == 2
    assert sessao.query(Exemplo).count() == 2


def test_importar_csv_ignora_texto_repetido(sessao, tmp_path):
    """A constraint UNIQUE existe justamente porque o CSV atual aceita repetido."""
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text("comentario,label_final\nte odeio,1\nte odeio,1\n", encoding="utf-8")
    quantos = seed.importar_csv(sessao, csv, usuario.id)
    assert quantos == 1


def test_main_usuario_existente_importa_csv_sem_pedir_senha(sessao, tmp_path, monkeypatch):
    """python -m app.seed --importar-csv, o comando do docstring do módulo, roda de
    novo para um usuário já existente sem travar pedindo senha nenhuma.
    """
    seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text("comentario,label_final\nte odeio,1\n", encoding="utf-8")

    monkeypatch.setattr(seed, "SessionLocal", lambda: sessao)
    monkeypatch.setattr(
        seed, "getpass", lambda *a, **k: pytest.fail("getpass não devia ser chamado")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["app.seed", "--email", "eu@exemplo.com", "--importar-csv", str(csv)],
    )

    seed.main()

    assert sessao.query(Exemplo).count() == 1


def test_main_usuario_existente_gerar_senha_nao_imprime_senha(sessao, monkeypatch, capsys):
    """--gerar-senha para um usuário que já existe (sem --resetar-senha) não deve
    gerar nem mostrar senha nenhuma: nada foi gravado para ela mostrar.
    """
    seed.criar_usuario(sessao, "eu@exemplo.com", "abc")

    monkeypatch.setattr(seed, "SessionLocal", lambda: sessao)
    monkeypatch.setattr("sys.argv", ["app.seed", "--gerar-senha", "--email", "eu@exemplo.com"])

    seed.main()

    saida = capsys.readouterr().out
    assert "aparece só esta vez" not in saida
    assert "já existe" in saida


def test_main_usuario_novo_gerar_senha_imprime_a_senha_real(sessao, monkeypatch, capsys):
    """A senha impressa para um usuário novo precisa ser a mesma que foi gravada —
    não só uma mensagem bonita.
    """
    monkeypatch.setattr(seed, "SessionLocal", lambda: sessao)
    monkeypatch.setattr("sys.argv", ["app.seed", "--gerar-senha", "--email", "novo@exemplo.com"])

    seed.main()

    saida = capsys.readouterr().out
    linha_senha = next(linha for linha in saida.splitlines() if "aparece só esta vez" in linha)
    senha_impressa = linha_senha.split(": ", 1)[1]

    usuario = sessao.scalar(select(Usuario).where(Usuario.email == "novo@exemplo.com"))
    assert usuario is not None
    assert security.conferir_senha(senha_impressa, usuario.senha_hash)
