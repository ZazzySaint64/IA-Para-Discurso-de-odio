"""Sobe o projeto inteiro com um comando só: cria o .env se faltar, aplica as
migrations, sobe a API e o Streamlit, e abre o navegador. Ctrl+C encerra tudo.

Motivo de existir: o README pedia dois terminais e um `printf` que não existe
no PowerShell. `python main.py` funciona igual em PowerShell, cmd e Git Bash.
"""

import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from secrets import token_urlsafe

RAIZ = Path(__file__).resolve().parent
ENV = RAIZ / ".env"
PORTA_API = 8000
PORTA_STREAMLIT = 8501


def checar_portas() -> None:
    for porta in (PORTA_API, PORTA_STREAMLIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", porta)) == 0:
                sys.exit(f"Porta {porta} já está em uso. Pare o que estiver nela e rode de novo.")


def criar_env_se_faltar() -> dict[str, str]:
    if ENV.exists():
        print(f".env já existe em {ENV}, deixei como está.")
    else:
        ENV.write_text(
            "DATABASE_URL=sqlite:///./dev.db\n"
            f"JWT_SECRET={token_urlsafe(32)}\n"
            "MODELO_PATH=ml/artefatos/modelo.pkl\n"
            "TREINO_HABILITADO=true\n",
            encoding="utf-8",
        )
        print(f".env criado em {ENV}")
    linhas = ENV.read_text(encoding="utf-8").splitlines()
    partes = (linha.split("=", 1) for linha in linhas if "=" in linha)
    return {chave.strip(): valor.strip() for chave, valor in partes}


def aplicar_migrations() -> None:
    print("Aplicando migrations...")
    if subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=RAIZ).returncode:
        sys.exit("Falha ao aplicar migrations (veja o erro acima).")


def checar_usuario(env: dict[str, str]) -> None:
    # Só sabemos checar SQLite local sem puxar SQLAlchemy pra dentro deste script.
    db_url = env.get("DATABASE_URL", "")
    if not db_url.startswith("sqlite"):
        return
    caminho = RAIZ / db_url.split("///", 1)[-1]
    if not caminho.exists():
        return
    conexao = sqlite3.connect(caminho)
    try:
        tem_usuario = conexao.execute("SELECT COUNT(*) FROM usuario").fetchone()[0] > 0
    finally:
        conexao.close()
    if not tem_usuario:
        print(
            "Nenhum usuário cadastrado ainda. Sem um, o painel de treino do Streamlit fica "
            "inutilizável (login sempre falha). Rode em outro terminal:\n"
            "    python -m app.seed --gerar-senha"
        )


def esperar(url: str, processo: subprocess.Popen, nome: str, exigir_200: bool = True) -> None:
    """Sem redirecionar stdout/stderr, o log do processo já aparece ao vivo
    neste terminal — por isso a falha abaixo manda olhar pra cima, em vez de
    tentar recapturar algo que já foi impresso.

    Poll em vez de sleep(N): o tempo de start varia (carregar o modelo,
    compilar bytecode etc.), e um valor fixo ou atrasa à toa ou falha cedo."""
    inicio = time.time()
    while time.time() - inicio < 60:
        if processo.poll() is not None:
            sys.exit(f"{nome} terminou sozinho (código {processo.returncode}). Veja o log acima.")
        try:
            resposta = urllib.request.urlopen(url, timeout=2)
            if not exigir_200 or resposta.status == 200:
                return
        except urllib.error.HTTPError:
            if not exigir_200:
                return
        except OSError:
            pass
        time.sleep(1)
    sys.exit(f"{nome} não respondeu em {url} depois de 60s. Veja o log acima.")


def main() -> None:
    checar_portas()
    env = criar_env_se_faltar()
    aplicar_migrations()
    checar_usuario(env)

    api = streamlit = None
    try:
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORTA_API)], cwd=RAIZ
        )
        esperar(f"http://localhost:{PORTA_API}/health", api, "api")

        streamlit = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/app.py",
                "--server.port",
                str(PORTA_STREAMLIT),
                "--server.headless",
                "true",
            ],
            cwd=RAIZ,
        )
        esperar(f"http://localhost:{PORTA_STREAMLIT}", streamlit, "streamlit", exigir_200=False)

        webbrowser.open(f"http://localhost:{PORTA_STREAMLIT}")
        print(
            f"\nAPI:       http://localhost:{PORTA_API}  (docs em /docs)\n"
            f"Streamlit: http://localhost:{PORTA_STREAMLIT}\n"
            "Ctrl+C encerra os dois.\n"
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        for processo in (api, streamlit):
            if processo is not None:
                processo.terminate()
        for processo in (api, streamlit):
            if processo is None:
                continue
            try:
                processo.wait(timeout=5)
            except subprocess.TimeoutExpired:
                processo.kill()


if __name__ == "__main__":
    main()
