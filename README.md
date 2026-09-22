# IA para Discurso de Ódio

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)
![Postgres](https://img.shields.io/badge/Postgres-16-336791)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio/actions/workflows/ci.yml/badge.svg)](https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio/actions/workflows/ci.yml)

> **EN:** A Portuguese hate-speech classifier (TF-IDF + Logistic Regression) served as a REST API — FastAPI, Postgres, JWT auth, background retraining, Docker, CI — with a Streamlit frontend that is a pure HTTP client. Full details below are in Portuguese, the project's language.

## No ar

**`<< ainda não publiquei — troque esta linha por https://SEU-APP.onrender.com quando o deploy no Render estiver feito >>`**

Swagger (documentação interativa, com botão **Authorize**): `<url-acima>/docs`

O plano gratuito do Render hiberna o serviço depois de 15 minutos sem uso. Se o link estiver "dormindo", a primeira requisição demora cerca de 50 segundos pra acordar o container — as seguintes voltam ao normal.

![Swagger com o botão Authorize](docs/swagger.jpg)

## O que é

Uma API que classifica um comentário em português como discurso de ódio ou não. O modelo é TF-IDF + Regressão Logística, validado com 5-fold cross-validation, F1 macro ≈ 0,75 no dataset combinado (HateBR + ToLD-BR). O domínio (classificação de texto) não é o foco do projeto — o foco é a engenharia em volta dele: API REST documentada, banco relacional, autenticação, testes, CI e deploy.

Existe uma rota pública para classificar texto e consultar métricas, e rotas autenticadas por JWT para quem treina o modelo: registrar exemplos rotulados e disparar um retreino. O Streamlit em `frontend/` é só um cliente HTTP dessa API, sem lógica própria — a mesma separação que motivou reescrever o projeto.

![Painel de treino do Streamlit, tela de login](docs/streamlit_painel_treino.jpg)

## Arquitetura

![Arquitetura](docs/arquitetura.svg)

Regra central: **`app/` nunca importa `ml/`** — a API só lê o artefato treinado (`ml/artefatos/modelo.pkl`); quem treina é um processo separado, porque inferência e treino têm perfis de recurso opostos e essa separação é o que permite rodar a API em um plano gratuito de 512 MB.

```
app/          FastAPI: rotas, auth, banco, carregamento do modelo
ml/           treino: junta os datasets, valida com cross-validation, escreve modelo.pkl
frontend/     Streamlit, só chamadas HTTP pra API
tests/        pytest + httpx, SQLite em memória
alembic/      migrations
```

## Endpoints

| Método | Rota | Auth | Sucesso | Erros |
|---|---|---|---|---|
| `POST` | `/auth/login` | — | `200` + token | `401` |
| `POST` | `/predicoes` | — | `201` | `422`, `503`, `429` |
| `GET` | `/predicoes` | JWT | `200` paginado | `401` |
| `POST` | `/exemplos` | JWT | `201` | `409`, `422`, `401` |
| `GET` | `/exemplos` | JWT | `200` paginado | `401` |
| `POST` | `/treinos` | JWT | `202` + id | `409`, `503`, `401` |
| `GET` | `/treinos/{id}` | JWT | `200` | `404`, `401` |
| `GET` | `/metricas` | — | `200` | — |
| `GET` | `/health` | — | `200` | `503` |

## Como rodar

Pré-requisito: Docker e Docker Compose. Os comandos que dependem de Docker **não foram executados nesta máquina** (sem Docker instalado aqui) — `pytest` e `ruff`, mais abaixo, foram, de verdade, nesta mesma máquina.

```bash
git clone https://github.com/ZazzySaint64/IA-Para-Discurso-de-dio.git
cd IA-Para-Discurso-de-dio
docker compose up --build
```

Sobe Postgres, a API em `localhost:8000` (`/docs` pro Swagger) e o Streamlit em `localhost:8501`. As migrations rodam sozinhas no início do container (`entrypoint.sh`).

Criar o usuário que treina o modelo (a senha aparece uma vez no terminal, salva num gerenciador de senhas):

```bash
docker compose run --rm api python -m app.seed --gerar-senha
```

Retreinar o modelo roda **fora** do container — a imagem não leva os datasets, e treino e inferência têm perfis de recurso opostos (ver "Decisões de projeto"). Baixe o [HateBR](https://github.com/franciellevargas/HateBR) e o [ToLD-BR](https://github.com/JAugusto97/ToLD-Br) e coloque `HateBR.csv` e `ToLD-BR.csv` em `HateBR-7.0.0/dataset/`, depois, com o Postgres do `docker compose up` já no ar (ele expõe a porta 5432):

```bash
pip install -r requirements.txt
python -m ml.treinar
```

Só substitui `ml/artefatos/modelo.pkl` se o novo F1 for maior que o melhor já registrado. (Este comando específico não foi rodado ao escrever este README — rodá-lo de verdade reescreveria o artefato já versionado no repositório; a sintaxe foi conferida lendo `ml/treinar.py`, não executando-o.)

Rodar os testes (verificado nesta máquina, sem Docker, com SQLite em memória):

```bash
pip install -r requirements-dev.txt
pytest --cov=app --cov-report=term-missing
```

61 testes, cobertura de 94% em `app/`.

## Como colocar no ar

O deploy é via [Render](https://render.com) Blueprint, lendo o `render.yaml` já commitado no repositório:

1. Cria uma conta no Render.
2. **New → Blueprint**, conecta este repositório do GitHub.
3. O Render lê o `render.yaml` e cria sozinho o banco (`hatebr-db`, Postgres, plano free) e o serviço web (`hatebr-api`, Docker, plano free, health check em `/health`), gerando o `JWT_SECRET` automaticamente. `TREINO_HABILITADO` já entra como `false` em produção.
4. Espera o build. As migrations rodam sozinhas no start do container (`entrypoint.sh`).
5. Roda o seed **uma vez**, contra o banco do Render: no dashboard do `hatebr-db`, aba **Connect**, copia a "External Database URL", e roda localmente:
   ```bash
   DATABASE_URL="<external-database-url-do-render>" python -m app.seed --gerar-senha
   ```
   Guarda a senha impressa — ela não é gravada em lugar nenhum, só aparece essa vez.

## Decisões de projeto

- **SHA-256 em variável de ambiente virou bcrypt no banco.** SHA-256 é rápido de propósito, o oposto do que se quer em hash de senha; bcrypt é lento de propósito e já embute salt, então a mesma senha gera hashes diferentes e rainbow table não serve.
- **CSV, JSON e log soltos viraram tabelas.** Os exemplos ensinados, a melhor métrica já alcançada e o histórico de execuções, antes um CSV, um JSON e um arquivo de log soltos, viraram as tabelas `exemplo` e `treino` no Postgres. A constraint `UNIQUE` em `exemplo.texto` impede o mesmo exemplo repetido, coisa que o CSV aceitava e enviesava o treino — a API devolve `409`.
- **Modelo e vetorizador separados viraram um pipeline só.** Salvar os dois passos (`TfidfVectorizer` + `LogisticRegression`) separadamente permite que saiam de sincronia; agora é um `sklearn.Pipeline` único, `joblib.dump` de um arquivo só.
- **Treino separado da API.** Perfis de recurso opostos — inferência quer latência baixa e pouca RAM, treino quer bastante RAM por alguns minutos — e essa separação é o que cabe em um plano gratuito de 512 MB.
- **Retreino desligado em produção, com o motivo, não escondido.** A flag `TREINO_HABILITADO` (padrão `false` em produção) controla `POST /treinos`; sem ela a rota devolve `503` explicando exatamente por quê (RAM insuficiente pra cross-validation, imagem sem os datasets).
- **Predição não guarda IP nem identificação de quem digitou.** A tabela `predicao` só tem texto, label e confiança — decisão pensando em LGPD, já que a rota é pública.

## O que ficou de fora e por quê

- **Frontend em React ou Next.js.** O Streamlit como cliente HTTP já demonstra a separação de camadas dentro do prazo do projeto.
- **Fila de tarefas (Celery, Redis, RQ).** `BackgroundTasks` do FastAPI resolve o caso de um único usuário treinando o modelo; fila seria complexidade sem demanda.
- **Troca do modelo por transformers (BERTimbau).** O ganho seria de domínio (melhor F1), não de engenharia, e o modelo não cabe em 512 MB de RAM.
- **Cadastro público de usuários, recuperação de senha, papéis e permissões.** Só o dono treina o modelo; usuários nascem por `app/seed.py`, não por rota pública — menos superfície de ataque, menos código.
- **Observabilidade além de logs estruturados.** Fora de escopo para um projeto de portfólio de duas semanas.

## Limitações conhecidas

- **`POST /treinos` tem uma corrida (TOCTOU).** A checagem de "já existe treino rodando" e a criação do novo registro não são atômicas: duas requisições verdadeiramente simultâneas podem, em teoria, iniciar dois treinos ao mesmo tempo. O caso realista — um duplo clique no botão do painel — está coberto, porque a linha do primeiro treino é commitada no banco antes da resposta voltar pro cliente.
- **Retreino desligado no container e em produção.** `TREINO_HABILITADO=false` tanto no `docker compose` local quanto no Render; para retreinar de fato, roda `python -m ml.treinar` fora do Docker (ver "Como rodar"). É uma limitação deliberada, não um bug — está documentada em "Decisões de projeto".
- **Qualidade real do modelo: F1 macro ≈ 0,75.** É um modelo linear simples (TF-IDF + Regressão Logística), não um transformer. Ele erra frases curtas e sem alvo explícito — por exemplo, classifica "eu te odeio" como não-ódio, com confiança baixa (~60%). Serve bem como demonstração de engenharia; não é production-grade para moderação de conteúdo real.

## Licenças dos datasets

Modelo treinado com dois datasets acadêmicos de português. Eles **não** são redistribuídos neste repositório — cada um tem sua própria licença:

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

Licença do código: MIT (ver `LICENSE`).
