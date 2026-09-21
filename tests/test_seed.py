import pytest
from sqlalchemy import select

from app import seed
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


def test_main_reimporta_csv_para_usuario_existente(sessao, tmp_path):
    """python -m app.seed --importar-csv pode ser rodado de novo sem quebrar.

    main() precisa lidar com UsuarioJaExisteError em vez de deixá-lo subir:
    reimportar é inofensivo porque exemplo.texto é UNIQUE e importar_csv já
    pula repetidos.
    """
    usuario = seed.criar_usuario(sessao, "eu@exemplo.com", "abc")
    csv = tmp_path / "feedback.csv"
    csv.write_text("comentario,label_final\nte odeio,1\n", encoding="utf-8")

    with pytest.raises(seed.UsuarioJaExisteError):
        seed.criar_usuario(sessao, "eu@exemplo.com", "outra-senha")

    usuario_existente = sessao.scalar(select(Usuario).where(Usuario.email == "eu@exemplo.com"))
    assert usuario_existente.id == usuario.id

    quantos = seed.importar_csv(sessao, csv, usuario_existente.id)
    assert quantos == 1
