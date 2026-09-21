import pandas as pd
import pytest

from ml import dados


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
