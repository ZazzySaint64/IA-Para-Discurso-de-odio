from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models import Treino


@pytest.fixture
def treino_falso(monkeypatch):
    """Treinar de verdade leva minutos. O teste verifica o fluxo, não o sklearn."""
    from app.routers import treinos

    monkeypatch.setattr(
        treinos,
        "_executar_treino",
        lambda treino_id, db_factory: None,
    )


def test_criar_treino_exige_token(cliente):
    assert cliente.post("/treinos").status_code == 401


def test_criar_treino_devolve_202_e_id(cliente_logado, treino_falso):
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 202
    assert "id" in resposta.json()


def test_criar_treino_com_um_ja_rodando_devolve_409(cliente_logado, sessao, treino_falso):
    sessao.add(Treino(status="rodando"))
    sessao.commit()
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 409


def test_consultar_treino_inexistente_devolve_404(cliente_logado):
    assert cliente_logado.get("/treinos/999").status_code == 404


def test_consultar_treino_devolve_status(cliente_logado, sessao):
    treino = Treino(status="concluido", f1_macro=0.75, desvio=0.01, qtd_exemplos=100)
    sessao.add(treino)
    sessao.commit()
    resposta = cliente_logado.get(f"/treinos/{treino.id}")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "concluido"
    assert resposta.json()["f1_macro"] == 0.75


def test_treino_desligado_devolve_503(cliente_logado, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "TREINO_HABILITADO", False)
    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 503
    assert "desligado" in resposta.json()["detail"].lower()


def test_treino_com_erro_no_meio_ainda_consegue_gravar_falhou(sessao, monkeypatch):
    """Regressão: um erro de banco no meio do treino deixa a sessão suja
    (flush falhou). Sem um rollback() antes de gravar o registro de erro, o
    commit do próprio bloco except também falha com PendingRollbackError, e o
    treino fica preso em "rodando" para sempre, sem nada registrado.
    """
    import ml.treinar as ml_treinar
    from app.models import Usuario
    from app.routers import treinos

    treino = Treino(status="rodando")
    sessao.add(treino)
    sessao.commit()
    tid = treino.id

    sessao.add(Usuario(email="dup@exemplo.com", senha_hash="a"))
    sessao.commit()

    def treinar_que_suja_a_sessao(db=None, **kwargs):
        db.add(Usuario(email="dup@exemplo.com", senha_hash="b"))  # viola unique
        db.commit()

    monkeypatch.setattr(ml_treinar, "treinar", treinar_que_suja_a_sessao)

    treinos._executar_treino(tid, lambda: sessao)

    atualizado = sessao.get(Treino, tid)
    assert atualizado.status == "falhou"
    assert atualizado.erro


def test_treino_travado_ha_mais_de_uma_hora_libera_novo_treino(
    cliente_logado, sessao, treino_falso
):
    antigo = Treino(status="rodando", iniciado_em=datetime.now(UTC) - timedelta(hours=2))
    sessao.add(antigo)
    sessao.commit()

    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 202

    assert antigo.status == "falhou"
    assert antigo.erro


def test_treino_travado_ha_pouco_tempo_ainda_devolve_409(cliente_logado, sessao, treino_falso):
    recente = Treino(status="rodando", iniciado_em=datetime.now(UTC) - timedelta(minutes=5))
    sessao.add(recente)
    sessao.commit()

    resposta = cliente_logado.post("/treinos")
    assert resposta.status_code == 409
    assert recente.status == "rodando"


@pytest.mark.parametrize("substituiu, chamadas_esperadas", [(True, 1), (False, 0)])
def test_treino_recarrega_o_modelo_so_quando_o_artefato_muda(
    sessao, monkeypatch, tmp_path, substituiu, chamadas_esperadas
):
    """Sem recarregar, a API segue respondendo com o .pkl antigo até reiniciar.

    O treino é falso de propósito: o que está sendo testado é a religação do
    pipeline, não o sklearn.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    import ml.treinar as ml_treinar
    from app.config import settings
    from app.routers import treinos

    artefato = tmp_path / "modelo.pkl"
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(["eu te odeio", "bom dia"], [1, 0])
    joblib.dump(pipeline, artefato)
    monkeypatch.setattr(settings, "MODELO_PATH", str(artefato))

    def treinar_falso(db=None, **kwargs):
        return {
            "f1_macro": 0.8,
            "desvio": 0.01,
            "seed": 1,
            "qtd_exemplos": 10,
            "substituiu": substituiu,
        }

    monkeypatch.setattr(ml_treinar, "treinar", treinar_falso)

    recarregados = []
    monkeypatch.setattr(treinos.ml, "carregar_modelo", recarregados.append)

    treino = Treino(status="pendente")
    sessao.add(treino)
    sessao.commit()

    treinos._executar_treino(treino.id, lambda: sessao)

    atualizado = sessao.get(Treino, treino.id)
    assert atualizado.status == "concluido"
    assert atualizado.substituiu is substituiu
    assert len(recarregados) == chamadas_esperadas
    if chamadas_esperadas:
        assert Path(recarregados[0]) == artefato


def test_consultar_treino_expoe_substituiu(cliente_logado, sessao):
    """O painel precisa distinguir "melhorou" de "manteve" sem adivinhar
    pelo F1: o front compara o F1 antes/depois, mas só este campo diz se o
    artefato em disco realmente mudou."""
    treino = Treino(status="concluido", f1_macro=0.75, substituiu=False)
    sessao.add(treino)
    sessao.commit()
    resposta = cliente_logado.get(f"/treinos/{treino.id}")
    assert resposta.json()["substituiu"] is False
