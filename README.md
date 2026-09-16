# IA para Discurso de Ódio

Uma IA em treinamento pra detectar se uma frase é discurso de ódio ou não. A ideia é simples: eu mostro exemplos pra ela, digo se é ódio ou não, e ela vai aprendendo com isso. Só eu tenho acesso a esse treinamento, pra não deixar qualquer um ensinando errado pro modelo.

## Antes de rodar

Esse repositório tem só o código. Não subi dataset, modelo treinado nem os rótulos que eu fui dando (tá tudo no `.gitignore`), datasets acadêmicos têm licença própria, e os rótulos são coisa minha mesmo.

Pra rodar do zero:

1. `pip install -r requirements.txt`
2. Baixa o [HateBR](https://github.com/franciellevargas/HateBR) e o [ToLD-BR](https://github.com/JAugusto97/ToLD-Br) (cada um com sua licença, dá uma lida antes) e joga em `HateBR-7.0.0/dataset/HateBR.csv` e `HateBR-7.0.0/dataset/ToLD-BR.csv`.
3. Roda `python retreinar_modelo.py` uma vez pra gerar o `modelo_logistic_regression.pkl` e o `vetorizador.pkl`.

## As duas telas

**`classificador.py`** é a parte pública: qualquer um digita um comentário e vê o que o modelo acha.
```
python -m streamlit run classificador.py
```

**`painel_treino.py`** é onde eu ensino o modelo. Fica atrás de senha porque só eu devo mexer nisso.
```
setx HATEBR_SENHA_TREINO "sua_senha"
```
(fecha e abre o terminal de novo pra pegar a senha, aí roda)
```
python -m streamlit run painel_treino.py
```
Lá dentro: escrevo um comentário, escolho se é ódio ou não, clico em "Guardar este exemplo". Vou fazendo isso quantas vezes quiser e, quando achar que já deu, clico em "Atualizar modelo agora", ele retreina com tudo que eu ensinei até ali.

Também tem o `retreinar.bat`, um atalho de duplo clique que roda o retreino e guarda o log.

## O resto do código (não roda sozinho, é usado pelos dois de cima)

- `dados_treino.py`, junta o HateBR com o ToLD-BR e, se eu já tiver ensinado alguma coisa, soma os meus exemplos também.
- `retreinar_modelo.py`, treina do zero com validação cruzada. Só troca o modelo salvo se o novo for melhor.
- `testar_modelo.py`, teste rápido, só pra conferir que os `.pkl` não estão corrompidos.
- `test_dados_treino.py`, checagem automática de que os dados estão carregando certo.

## O que fica de fora do repositório (gerado na hora, não mexe à mão)

- `modelo_logistic_regression.pkl` e `vetorizador.pkl`, o modelo treinado e o que converte texto em número pra ele entender.
- `best_score.json`, guarda a melhor nota já alcançada, pra saber se vale a pena trocar o modelo.
- `feedback_treino.csv`, todos os exemplos que eu já ensinei.
- `historico_execucoes.log`, log de cada retreino.
- `HateBR-7.0.0/`, os datasets originais.
