def test_predicao_caminho_feliz(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "eu te odeio"})
    assert resposta.status_code == 201
    corpo = resposta.json()
    assert corpo["label"] == 1
    assert corpo["rotulo"] == "Discurso de Ódio"
    assert corpo["confianca"] == 0.87


def test_predicao_texto_vazio(cliente):
    resposta = cliente.post("/predicoes", json={"texto": ""})
    assert resposta.status_code == 422
    assert "detail" in resposta.json()


def test_predicao_texto_so_espaco(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "   "})
    assert resposta.status_code == 422


def test_predicao_texto_longo_demais(cliente):
    resposta = cliente.post("/predicoes", json={"texto": "a" * 1001})
    assert resposta.status_code == 422


def test_predicao_sem_modelo_devolve_503(cliente, monkeypatch):
    from app import ml

    def indisponivel(texto):
        raise ml.ModeloIndisponivelError()

    monkeypatch.setattr(ml, "prever", indisponivel)
    resposta = cliente.post("/predicoes", json={"texto": "oi"})
    assert resposta.status_code == 503


def test_predicao_e_gravada_no_banco(cliente, sessao):
    from app.models import Predicao

    cliente.post("/predicoes", json={"texto": "eu te odeio"})
    gravadas = sessao.query(Predicao).all()
    assert len(gravadas) == 1
    assert gravadas[0].texto == "eu te odeio"
    assert gravadas[0].label == 1
