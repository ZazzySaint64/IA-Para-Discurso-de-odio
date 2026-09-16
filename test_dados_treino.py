from dados_treino import carregar_dados

if __name__ == "__main__":
    dados = carregar_dados(incluir_feedback=False)
    assert list(dados.columns) == ["comentario", "label_final"]
    assert len(dados) > 0
    assert set(dados["label_final"].unique()) <= {0, 1}
    print(f"OK: {len(dados)} exemplos carregados.")
