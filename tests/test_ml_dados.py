from pathlib import Path

import pandas as pd
import pytest

from ml import dados

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_TREINO_CURADO = RAIZ / "ml" / "frases_curtas.csv"
CAMINHO_AVALIACAO_CURADA = RAIZ / "ml" / "avaliacao_frases_curtas.csv"


@pytest.fixture
def pasta_datasets(tmp_path):
    pd.DataFrame({"comentario": ["te odeio", "bom dia"], "label_final": [1, 0]}).to_csv(
        tmp_path / "HateBR.csv", index=False
    )
    pd.DataFrame(
        {
            "text": ["some morto", "que dia lindo"],
            "homophobia": [1, 0],
            "obscene": [1, 0],
            "insult": [0, 0],
            "racism": [0, 0],
            "misogyny": [0, 0],
            "xenophobia": [0, 0],
        }
    ).to_csv(tmp_path / "ToLD-BR.csv", index=False)
    return tmp_path


def test_carregar_dados_junta_os_dois_datasets(pasta_datasets):
    df = dados.carregar_dados(db=None, pasta_datasets=pasta_datasets)
    assert list(df.columns) == ["comentario", "label_final"]
    assert len(df) == 4
    assert set(df["label_final"].unique()) <= {0, 1}


def test_told_br_vira_odio_com_duas_categorias(pasta_datasets):
    """A regra de rotulagem do ToLD-BR: 2 ou mais categorias de toxicidade = ódio."""
    df = dados.carregar_dados(db=None, pasta_datasets=pasta_datasets)
    assert df[df["comentario"] == "some morto"]["label_final"].iloc[0] == 1
    assert df[df["comentario"] == "que dia lindo"]["label_final"].iloc[0] == 0


def test_carregar_dados_inclui_exemplos_do_banco(pasta_datasets, sessao, usuario):
    from app.models import Exemplo

    sessao.add(Exemplo(texto="frase ensinada", label=1, usuario_id=usuario.id))
    sessao.commit()
    df = dados.carregar_dados(db=sessao, pasta_datasets=pasta_datasets)
    assert len(df) == 5
    assert "frase ensinada" in df["comentario"].values


def test_dataset_faltando_levanta_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="HateBR.csv"):
        dados.carregar_dados(db=None, pasta_datasets=tmp_path)


def _normalizar(serie: pd.Series) -> set[str]:
    return set(serie.astype(str).str.strip().str.lower())


def test_frases_curtas_treino_e_avaliacao_nao_se_sobrepoem():
    """A avaliação só é honesta se nenhuma frase de treino vazar pra lá."""
    treino = pd.read_csv(CAMINHO_TREINO_CURADO)
    avaliacao = pd.read_csv(CAMINHO_AVALIACAO_CURADA)
    sobreposicao = _normalizar(treino["comentario"]) & _normalizar(avaliacao["comentario"])
    assert not sobreposicao, f"frases repetidas entre treino e avaliação: {sobreposicao}"


def test_carregar_dados_nao_le_o_csv_de_avaliacao(pasta_datasets):
    """ml/avaliacao_frases_curtas.csv não pode entrar no treino por nenhum caminho."""
    avaliacao = pd.read_csv(CAMINHO_AVALIACAO_CURADA)
    df = dados.carregar_dados(db=None, pasta_datasets=pasta_datasets)
    vazou = _normalizar(df["comentario"]) & _normalizar(avaliacao["comentario"])
    assert not vazou, f"frases de avaliação apareceram nos dados de treino: {vazou}"


def test_carregar_frases_curtas_devolve_o_csv_curado():
    df = dados.carregar_frases_curtas()
    assert list(df.columns) == ["comentario", "label_final"]
    assert len(df) > 100
    assert set(df["label_final"].unique()) <= {0, 1}


def test_carregar_frases_curtas_com_caminho_inexistente_devolve_vazio(tmp_path):
    df = dados.carregar_frases_curtas(tmp_path / "nao_existe.csv")
    assert df.empty
    assert list(df.columns) == ["comentario", "label_final"]
