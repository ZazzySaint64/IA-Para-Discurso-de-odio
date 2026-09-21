"""Cria o usuário administrador e importa o feedback legado.

Substitui o gerar_hash_senha.py: a senha nunca vira variável de ambiente,
e com --gerar-senha ela nem chega a ser digitada.

Uso:
    python -m app.seed --gerar-senha
    python -m app.seed --email eu@exemplo.com
    python -m app.seed --resetar-senha --email eu@exemplo.com
    python -m app.seed --importar-csv feedback_treino.csv
"""

import argparse
import csv as csv_lib
import secrets
from getpass import getpass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import security
from app.database import SessionLocal
from app.models import Exemplo, Usuario


class UsuarioJaExisteError(RuntimeError):
    pass


def gerar_senha() -> str:
    """~120 bits de entropia, via CSPRNG do sistema operacional."""
    return secrets.token_urlsafe(15)


def criar_usuario(db: Session, email: str, senha: str) -> Usuario:
    if db.scalar(select(Usuario).where(Usuario.email == email)) is not None:
        raise UsuarioJaExisteError(f"Já existe usuário com o email {email}")
    usuario = Usuario(email=email, senha_hash=security.gerar_hash(senha))
    db.add(usuario)
    db.commit()
    return usuario


def resetar_senha(db: Session, email: str, senha: str) -> Usuario:
    usuario = db.scalar(select(Usuario).where(Usuario.email == email))
    if usuario is None:
        raise ValueError(f"Não existe usuário com o email {email}")
    usuario.senha_hash = security.gerar_hash(senha)
    db.commit()
    return usuario


def importar_csv(db: Session, caminho: Path, usuario_id: int) -> int:
    """Importa feedback_treino.csv para a tabela exemplo, pulando repetidos."""
    ja_existem = {t for (t,) in db.execute(select(Exemplo.texto))}
    importados = 0
    with open(caminho, encoding="utf-8") as f:
        for linha in csv_lib.DictReader(f):
            texto = linha["comentario"].strip()
            if not texto or texto in ja_existem:
                continue
            db.add(Exemplo(texto=texto, label=int(linha["label_final"]), usuario_id=usuario_id))
            ja_existem.add(texto)
            importados += 1
    db.commit()
    return importados


def main() -> None:
    p = argparse.ArgumentParser(description="Cria o usuário admin e importa dados legados.")
    p.add_argument("--email", default="admin@hatebr.local")
    p.add_argument("--gerar-senha", action="store_true", help="gera uma senha forte")
    p.add_argument("--resetar-senha", action="store_true")
    p.add_argument("--importar-csv", type=Path, help="caminho do feedback_treino.csv")
    args = p.parse_args()

    db = SessionLocal()
    try:
        usuario = db.scalar(select(Usuario).where(Usuario.email == args.email))

        def obter_senha() -> str:
            if args.gerar_senha:
                return gerar_senha()
            return getpass("Senha (não aparece enquanto você digita): ")

        # None enquanto nenhuma senha for de fato gravada — controla se e o que é
        # impresso depois, para nunca mostrar uma senha que não foi salva.
        senha_nova = None

        if args.resetar_senha:
            if usuario is None:
                raise ValueError(f"Não existe usuário com o email {args.email}")
            senha_nova = obter_senha()
            usuario = resetar_senha(db, args.email, senha_nova)
            print(f"Senha trocada para {usuario.email}")
        elif usuario is None:
            senha_nova = obter_senha()
            usuario = criar_usuario(db, args.email, senha_nova)
            print(f"Usuário criado: {usuario.email}")
        else:
            print(f"Usuário {usuario.email} já existe, senha inalterada.")

        if senha_nova is not None and args.gerar_senha:
            print(f"Senha (aparece só esta vez, salve agora): {senha_nova}")

        if args.importar_csv:
            quantos = importar_csv(db, args.importar_csv, usuario.id)
            print(f"{quantos} exemplos importados de {args.importar_csv}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
