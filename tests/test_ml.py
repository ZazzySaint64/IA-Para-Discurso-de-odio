import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import ml


@pytest.fixture
def caminho_modelo(tmp_path):
    """Pipeline minúsculo, treinado na hora. Não depende do .pkl real."""
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(
        ["eu te odeio seu lixo", "vá morrer", "bom dia pessoal", "que dia lindo"],
        [1, 1, 0, 0],
    )
    caminho = tmp_path / "modelo.pkl"
    joblib.dump(pipeline, caminho)
    return caminho


def test_prever_sem_modelo_carregado_levanta_erro():
    ml._pipeline = None
    with pytest.raises(ml.ModeloIndisponivelError):
        ml.prever("qualquer coisa")


def test_carregar_modelo_inexistente_levanta_erro(tmp_path):
    with pytest.raises(FileNotFoundError):
        ml.carregar_modelo(tmp_path / "nao_existe.pkl")


def test_prever_devolve_label_e_confianca(caminho_modelo):
    ml.carregar_modelo(caminho_modelo)
    label, confianca = ml.prever("eu te odeio seu lixo")
    assert label in (0, 1)
    assert 0.0 <= confianca <= 1.0


def test_modelo_carregado_reflete_o_estado(caminho_modelo):
    ml._pipeline = None
    assert ml.modelo_carregado() is False
    ml.carregar_modelo(caminho_modelo)
    assert ml.modelo_carregado() is True
