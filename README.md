# IA para Discurso de Ódio

[![CI](https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio/actions/workflows/ci.yml/badge.svg)](https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)

> **EN:** A Portuguese hate-speech classifier served as a REST API (FastAPI, Postgres, JWT auth, background retraining, Docker, CI) with a Streamlit frontend that only talks to the API over HTTP. Live at https://hatebr-web.onrender.com. Details below are in Portuguese, the project's language.

Um site onde você escreve um comentário em português e ele te diz se aquilo é ofensivo ou não. Por trás tem um modelo que eu treinei com uns 28 mil comentários reais de redes sociais, e um painel com senha onde eu ensino exemplos novos pra ele.

## Testa aqui

**https://hatebr-web.onrender.com**

Tá num servidor gratuito, que dorme depois de 15 minutos sem ninguém usar. Se for o primeiro acesso do dia, a primeira resposta pode levar até um minuto e meio. Não tá quebrado, tá acordando. Depois disso fica rápido.

O que dá pra fazer em 30 segundos, na aba **Classificar**:

- Escreve "some da minha frente agora" e depois "some com essa dor de cabeça logo". A palavra é a mesma, mas só a primeira é contra alguém. O modelo acerta as duas, e nenhuma delas tava no que eu usei pra treinar ele.
- Faz o mesmo com "odeio quando a impressora trava". Tem "odeio", mas não é contra ninguém.
- Cola um comentário qualquer que você viu por aí e vê a porcentagem de confiança.
- Tenta enganar ele. Tem frase que ainda passa, eu listo as que eu conheço lá embaixo.

A aba **Painel de treino** pede login, é onde eu (só eu) ensino exemplos novos.

Pra quem é da área e quer mexer na API direto: https://hatebr-api.onrender.com/docs

## Feito com

- Python, a linguagem do projeto todo
- FastAPI, pra API
- PostgreSQL, o banco de dados
- SQLAlchemy e Alembic, pra falar com o banco e versionar as tabelas
- JWT e bcrypt, pro login do painel
- scikit-learn, pro modelo
- Streamlit, pra interface do site
- Docker, pra empacotar tudo
- GitHub Actions, pra rodar os testes sozinho a cada commit
- Render, onde tá hospedado

## O que eu aprendi fazendo isso

**Passar nos testes não é a mesma coisa que funcionar.** O primeiro deploy quebrou na hora. O Render me entrega o endereço do banco num formato (`postgresql://`) que o meu código não esperava, e o container caía antes de subir. Tudo passava nos testes e no CI, porque nenhum teste usava o endereço do jeito que o Render manda. Só apareceu em produção. Hoje tem teste pra esse formato.

**O login contava quais emails existiam.** Numa revisão de segurança apareceu que o login respondia mais rápido quando o email não estava cadastrado, porque nem chegava a conferir a senha. Medindo o tempo de resposta, dava pra descobrir quem tinha conta. A correção foi conferir a senha sempre, contra um hash falso quando o email não existe, pra resposta levar o mesmo tempo nos dois casos.

**O site mostrava "502" pra primeira pessoa que entrava.** O servidor gratuito dorme, e eu achei que a interface chamando a API ia acordar ela. Não acorda: o Render só acorda um serviço com tráfego vindo de fora, e a interface fala com a API por dentro. Fiquei um tempo com um retry esperando um boot que nunca começava, até olhar os logs e ver zero requisições chegando. A solução foi fazer o navegador de quem visita mandar um ping pra API logo que a página abre. Esse vem de fora, então acorda.

**"eu te odeio" saía como não ofensivo.** Os dados são quase todos comentários longos de política no Instagram. Quase não tem frase curta dirigida a uma pessoa, e "odeio" lá aparece mais em coisa tipo "odeio esse governo". Então eu escrevi um conjunto de frases curtas, inclusive umas traiçoeiras como "odeio segunda-feira" e "morri de rir", e separei outras 60 frases que o modelo nunca vê no treino, só pra medir. O acerto nessas 60 foi de 75% pra 92%. Separar essas 60 foi o que me deixou confiar no número, senão eu tava só medindo se ele decorou.

O projeto começou como dois scripts de Streamlit lendo o arquivo do modelo direto do disco. Virou um serviço de verdade, com a interface só conversando com a API. Essa virada foi o que eu mais aprendi.

---

Daqui pra baixo é a parte técnica, fechada em blocos. Clica pra abrir.

<details>
<summary><b>Arquitetura</b>: como as peças se conectam</summary>

![Arquitetura](docs/arquitetura.svg)

```
app/          FastAPI: rotas, auth, banco, carregamento do modelo
ml/           treino: junta os datasets, valida com cross-validation, escreve modelo.pkl
frontend/     Streamlit, só chamadas HTTP pra API
tests/        pytest + httpx, SQLite em memória
alembic/      migrations
main.py       sobe tudo local com um comando
```

A regra central é que `app/` não importa `ml/` em tempo de import. A API só lê o artefato treinado (`ml/artefatos/modelo.pkl`) com `joblib.load`. O único ponto de contato é um `from ml.treinar import treinar` adiado dentro de `_executar_treino` (`app/routers/treinos.py`), que só roda quando alguém pede um retreino. Isso não é só promessa: `tests/test_arquitetura.py` percorre todo `.py` de `app/` com `ast` e falha se alguém subir esse import pro topo do módulo.

O Streamlit em `frontend/app.py` não carrega modelo nem acessa banco. Ele só faz HTTP pra API. Quando a API tá dormindo, ele insiste por até 90 segundos de relógio antes de avisar que ela não respondeu, e dispara um `fetch` do navegador do visitante pro `/health` pra acordar o container.

O modelo é um `sklearn.Pipeline` com um `FeatureUnion` de dois TF-IDF (palavras de 1 a 2 e pedaços de 2 a 5 letras, `char_wb`) seguido de regressão logística. As 204 frases curtas que eu escrevi (`ml/frases_curtas.csv`) entram só no fit final, com peso 20, e nunca nas dobras da validação cruzada, pra não inflar o F1. As 60 frases de avaliação ficam em `ml/avaliacao_frases_curtas.csv` e nunca entram no treino (tem teste checando que os dois arquivos não se sobrepõem).

Números atuais:

- F1 macro na validação cruzada (5 dobras): 0,782, em 28 mil comentários (HateBR + ToLD-BR)
- Acerto nas 60 frases curtas nunca vistas: 92% (55 de 60). Antes era 75% (45 de 60)
- 85 testes, 96% de cobertura em `app/`

</details>

<details>
<summary><b>Rotas da API</b>: o que cada endpoint faz e devolve</summary>

| Método | Rota | Login | Sucesso | Erros |
|---|---|---|---|---|
| `POST` | `/auth/login` | não | `200` + token | `401` |
| `POST` | `/predicoes` | não | `201` | `422`, `503`, `429` |
| `GET` | `/predicoes` | JWT | `200` paginado | `401` |
| `POST` | `/exemplos` | JWT | `201` | `409`, `422`, `401` |
| `GET` | `/exemplos` | JWT | `200` paginado | `401` |
| `POST` | `/treinos` | JWT | `202` + id | `409`, `503`, `401` |
| `GET` | `/treinos/{id}` | JWT | `200` | `404`, `401` |
| `GET` | `/metricas` | não | `200` | |
| `GET` | `/health` | não | `200` | `503` |

`GET /` redireciona pro `/docs`. O `POST /predicoes` tem limite de 30 requisições por minuto por IP (`429` depois disso). O `/health` também confere o banco, não só se o processo tá vivo. No `/docs` tem o botão **Authorize** pra testar as rotas com login direto do navegador.

</details>

<details>
<summary><b>Decisões de projeto e por quê</b></summary>

- **Senha em bcrypt no banco, não SHA-256 em variável de ambiente.** Era assim na primeira versão. SHA-256 é rápido de propósito, o contrário do que se quer pra senha. bcrypt é lento de propósito e já vem com salt.
- **Login sempre confere um hash**, mesmo quando o email não existe, pra não vazar pelo tempo de resposta quais contas existem.
- **CSV, JSON e log soltos viraram tabelas.** Os exemplos ensinados e o histórico de treinos ficam nas tabelas `exemplo` e `treino`. Um `UNIQUE` em `exemplo.texto` barra exemplo repetido (a API devolve `409`), coisa que o CSV aceitava e enviesava o treino.
- **Modelo e vetorizador num arquivo só.** Salvar os dois separados deixava eles saírem de sincronia. Agora é um `Pipeline` único.
- **Treino separado da API.** Inferência quer pouca RAM e resposta rápida, treino quer muita RAM por alguns minutos. Separar é o que faz a API caber nos 512 MB do plano gratuito.
- **Retreino desligado em produção, e dizendo por quê.** Com `TREINO_HABILITADO=false` (no `render.yaml` e no `docker-compose.yml`), `POST /treinos` devolve `503` explicando o motivo: pouca RAM pra validação cruzada e imagem sem os datasets.
- **`JWT_SECRET` sem valor padrão.** Um default num repositório público seria um segredo que qualquer um lê. A aplicação prefere não subir a fingir que tá protegida. Em produção quem gera é o Render (`generateValue: true`).
- **Predição não guarda IP nem nada de quem digitou.** A tabela `predicao` só tem texto, resultado e confiança. A rota é pública, pensei na LGPD.
- **O retreino pela API só troca o modelo se o F1 melhorar** em relação ao melhor treino registrado.
- **Frases curtas com peso 20.** Testei pesos de 1 a 150. Acima de 20 o acerto nas frases curtas quase não sobe e o F1 nos dados originais começa a cair.

O que ficou de fora de propósito: frontend em React (o Streamlit já mostra a separação de camadas), fila tipo Celery (`BackgroundTasks` resolve pra um usuário só), transformers tipo BERTimbau (não cabe em 512 MB, e o ganho seria de modelo, não de engenharia), cadastro público de usuário (só eu treino, usuário nasce pelo `app/seed.py`), e observabilidade além de `logging`.

</details>

<details>
<summary><b>Como rodar</b>: um comando, ou as peças separadas</summary>

### O jeito rápido

```bash
git clone https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio.git
cd IA-Para-Discurso-de-dio
pip install -r requirements.txt -r frontend/requirements.txt
python main.py
```

O `main.py` cria o `.env` se ele não existir (com um `JWT_SECRET` novo e um SQLite local em `dev.db`), roda as migrations, sobe a API em `localhost:8000` e o Streamlit em `localhost:8501`, espera os dois responderem de verdade e abre o navegador. Ctrl+C fecha os dois. As portas 8000 e 8501 precisam estar livres, se não ele avisa qual tá ocupada e sai.

Pra usar o painel de treino precisa de um usuário (a senha aparece uma vez só no terminal):

```bash
python -m app.seed --gerar-senha
```

### Com Docker (Postgres de verdade)

Esses comandos eu não rodei na minha máquina, porque não tenho Docker instalado aqui. São os mesmos que o CI e o `docker-compose.yml` usam.

```bash
cp .env.exemplo .env
docker compose up --build
docker compose run --rm api python -m app.seed --gerar-senha
```

Sobe Postgres, a API e o Streamlit. As migrations rodam sozinhas no início do container (`entrypoint.sh`).

### Retreinar o modelo

Roda fora do container, porque a imagem não leva os datasets. Baixa o [HateBR](https://github.com/franciellevargas/HateBR) e o [ToLD-BR](https://github.com/JAugusto97/ToLD-Br), coloca `HateBR.csv` e `ToLD-BR.csv` em `HateBR-7.0.0/dataset/`, e com o `.env` criado:

```bash
python -m ml.treinar
```

Esse caminho sempre grava o `ml/artefatos/modelo.pkl` novo. A trava de "só troca se melhorar" vale pro retreino disparado pela API.

### Testes

```bash
pip install -r requirements-dev.txt
pytest --cov=app --cov-report=term-missing
```

85 testes, 96% de cobertura em `app/`. Rodam com SQLite em memória, sem Docker. No CI rodam contra um Postgres 16.

### Deploy

É um Blueprint do Render lendo o `render.yaml`: cria o banco `hatebr-db`, a API `hatebr-api` (Docker) e a interface `hatebr-web`, tudo no plano gratuito. Depois do primeiro deploy, o usuário é criado uma vez rodando o seed da minha máquina contra a URL externa do banco:

```bash
DATABASE_URL="<external-database-url-do-render>" python -m app.seed --gerar-senha
```

</details>

<details>
<summary><b>Limitações conhecidas</b>: o que ainda não funciona bem</summary>

- **O servidor gratuito dorme.** Depois de 15 minutos parado, o primeiro acesso leva até um minuto e meio.
- **Algumas frases curtas ainda passam.** Das 60 de avaliação, erra 5: "morre logo", "espero que se exploda", "eu te detesto" e "você me enoja" saem como não ofensivas (hostilidade sem palavrão e sem "odeio", tipo "detesto" e "enoja", o modelo ainda não pega), e "você é o máximo" sai como ofensiva. É um modelo linear simples, não um transformer. Serve pra mostrar a engenharia, não pra moderar conteúdo de verdade.
- **Os rótulos das frases curtas fui eu que escrevi.** Refletem o meu julgamento do que é ofensivo, e outra pessoa podia rotular diferente.
- **Dois retreinos exatamente ao mesmo tempo podem começar os dois.** A checagem de "já tem treino rodando" e a criação do novo não são atômicas. O duplo clique no botão tá coberto, porque o primeiro treino é gravado no banco antes da resposta voltar.
- **Se o modelo novo for gravado mas falhar ao carregar**, o treino fica como `falhou` sem guardar o F1, e o `/metricas` continua mostrando o F1 antigo. É raro (o arquivo precisa ser escrito certo e ler errado logo depois).
- **Retreino desligado em produção.** Pra retreinar tem que rodar `python -m ml.treinar` fora do Render.
- **O banco gratuito do Render expira depois de 30 dias.** Depois disso tem que recriar o banco e rodar o seed de novo.

</details>

<details>
<summary><b>Licenças dos datasets e créditos</b></summary>

O modelo foi treinado com dois datasets acadêmicos de português. Eles **não** estão neste repositório, cada um tem a sua licença, dá uma lida antes de usar:

- [HateBR](https://github.com/franciellevargas/HateBR)

  ```bibtex
  @inproceedings{vargas-etal-2022-hatebr,
      title = "{H}ate{BR}: A Large Expert Annotated Corpus of {B}razilian {I}nstagram Comments for Offensive Language and Hate Speech Detection",
      author = "Vargas, Francielle and Carvalho, Isabelle and Rodrigues de Góes, Fabiana and Pardo, Thiago and Benevenuto, Fabrício",
      booktitle = "Proceedings of the 13th Conference on Language Resources and Evaluation (LREC 2022)",
      year = "2022",
      url = "https://aclanthology.org/2022.lrec-1.777",
  }
  ```

- [ToLD-BR](https://github.com/JAugusto97/ToLD-Br)

  ```bibtex
  @inproceedings{leite2020toldbr,
      title = "Toxic Language Detection in Social Media for {B}razilian {P}ortuguese: New Dataset and Multilingual Analysis",
      author = "Leite, João Augusto and Silva, Diego Furtado and Bontcheva, Kalina and Scarton, Carolina",
      booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
      year = "2020",
  }
  ```

Licença do meu código: MIT (ver `LICENSE`).

</details>
