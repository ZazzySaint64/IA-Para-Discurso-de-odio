# IA para Discurso de Ódio

Uma IA em treinamento para detectar frases e analisar se é discurso de ódio ou não. Classifica um comentário em português como discurso de ódio ou não, e aprende com rótulos que você mesmo dá a ela.

## Antes de rodar

Este repositório tem só o código. Datasets, modelo treinado e rótulos ficam de fora (`.gitignore`) — veja por quê na tabela "Arquivos gerados automaticamente" abaixo.

1. `pip install -r requirements.txt`
2. Baixe os datasets [HateBR](https://github.com/franciellevargas/HateBR) e [ToLD-BR](https://github.com/JAugusto97/ToLD-Br) (cada um tem sua própria licença — leia antes de usar) e coloque em `HateBR-7.0.0/dataset/HateBR.csv` e `HateBR-7.0.0/dataset/ToLD-BR.csv`.
3. Rode `python retreinar_modelo.py` uma vez pra gerar `modelo_logistic_regression.pkl` e `vetorizador.pkl`.

## O que rodar

| Arquivo | O que é | Como rodar |
|---|---|---|
| `classificador.py` | Tela pública: qualquer um digita um comentário e vê o palpite do modelo. | `python -m streamlit run classificador.py` |
| `painel_treino.py` | Tela protegida por senha: só você ensina o modelo com novos exemplos e manda ele se atualizar. | Definir `HATEBR_SENHA_TREINO` antes, depois `python -m streamlit run painel_treino.py` |
| `retreinar.bat` | Atalho (duplo clique) que roda `retreinar_modelo.py` e guarda o log em `historico_execucoes.log`. | Duplo clique no arquivo |

## Arquivos de código (suporte, não roda direto)

| Arquivo | O que faz |
|---|---|
| `dados_treino.py` | Junta os dois datasets (HateBR + ToLD-BR) e, se existir, soma os exemplos que você ensinou em `painel_treino.py`. Usado por `retreinar_modelo.py` e `painel_treino.py`. |
| `retreinar_modelo.py` | Treina o modelo do zero com validação cruzada. Só salva o modelo novo se ele for melhor que o salvo em `best_score.json`. |
| `testar_modelo.py` | Teste rápido: carrega o modelo salvo e classifica uma frase de exemplo, só pra conferir que os arquivos `.pkl` não estão corrompidos. Usado por `classificador.py`. |
| `test_dados_treino.py` | Checagem automática de que `dados_treino.py` está carregando os dados direito. |

## Arquivos gerados automaticamente (não editar à mão)

| Arquivo | O que é |
|---|---|
| `modelo_logistic_regression.pkl` | O modelo treinado (o "cérebro" que decide ódio ou não). |
| `vetorizador.pkl` | Converte texto em números pro modelo entender. Tem que ser sempre o par do modelo acima. |
| `best_score.json` | Guarda a melhor nota (F1 macro) já alcançada, pra `retreinar_modelo.py` saber se vale a pena substituir o modelo. |
| `feedback_treino.csv` | Todos os exemplos que você ensinou em `painel_treino.py`, aguardando ou já usados no treino. |
| `historico_execucoes.log` | Log de cada vez que `retreinar.bat` rodou. |
| `HateBR-7.0.0/` | Pasta com os datasets originais (HateBR.csv e ToLD-BR.csv) usados no treino. Não mexer. |

## Como ensinar o modelo (resumo)

1. Defina a senha uma vez: `setx HATEBR_SENHA_TREINO "sua_senha"` (feche e abra o terminal depois).
2. Rode `painel_treino.py`, entre com a senha.
3. Escreva um comentário, escolha o rótulo certo, clique em "Guardar este exemplo".
4. Repita quantas vezes quiser, depois clique em "Atualizar modelo agora".
