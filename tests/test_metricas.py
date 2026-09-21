def test_metricas_com_banco_vazio(cliente):
    resposta = cliente.get("/metricas")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["total_predicoes"] == 0
    assert corpo["taxa_odio"] == 0.0
    assert corpo["total_exemplos"] == 0


def test_metricas_conta_predicoes_e_taxa(cliente):
    for _ in range(4):
        cliente.post("/predicoes", json={"texto": "eu te odeio"})
    resposta = cliente.get("/metricas")
    corpo = resposta.json()
    assert corpo["total_predicoes"] == 4
    # a fixture `cliente` faz prever() devolver sempre label 1
    assert corpo["taxa_odio"] == 1.0


def test_metricas_nao_exige_token(cliente):
    assert cliente.get("/metricas").status_code == 200
