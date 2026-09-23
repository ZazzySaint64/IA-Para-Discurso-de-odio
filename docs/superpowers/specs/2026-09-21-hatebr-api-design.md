# Design: HateBR como serviço (API REST + Postgres)

Data: 2026-09-21
Status: aprovado, aguardando plano de implementação

## Objetivo

Transformar o classificador de discurso de ódio, hoje dois scripts Streamlit que
carregam um `.pkl` direto do disco, em um serviço backend completo: API REST
documentada, banco relacional, autenticação com JWT, containers, testes
automatizados, CI e deploy público.

O objetivo é de portfólio: demonstrar competência de backend para vagas de
estágio. O domínio (classificação de texto) já está resolvido e não é o foco. O
foco é a engenharia em volta dele.

Critério de sucesso: um recrutador abre um link, vê o Swagger, faz login, testa
uma rota protegida e entende a arquitetura pelo README, tudo em menos de cinco
minutos.

## Contexto: o que existe hoje

| Arquivo | Papel atual | Destino |
|---|---|---|
| `classificador.py` | Streamlit público, carrega o modelo direto | Vira cliente HTTP da API |
| `painel_treino.py` | Streamlit com senha, salva feedback em CSV | Vira cliente HTTP da API |
| `dados_treino.py` | Junta HateBR + ToLD-BR + CSV de feedback | Vira `ml/dados.py`, lê do banco |
| `retreinar_modelo.py` | Treina com 5-fold CV, salva se melhorar | Vira `ml/treinar.py` |
| `testar_modelo.py` | Carrega os `.pkl` | Absorvido por `app/ml.py` |
| `test_dados_treino.py` | Asserts em `__main__` | Vira teste pytest |
| `gerar_hash_senha.py` | SHA-256 da senha para env var | Substituído por `app/seed.py` (bcrypt) |
| `feedback_treino.csv` | Exemplos ensinados | Tabela `exemplo` |
| `best_score.json` | Melhor F1 alcançado | Tabela `treino` |
| `historico_execucoes.log` | Log de cada retreino | Tabela `treino` |
| `modelo_*.pkl` + `vetorizador.pkl` | Dois artefatos separados | Um `ml/artefatos/modelo.pkl` |

Modelo atual: TF-IDF + Regressão Logística, F1 macro ~0,75 em 5-fold CV sobre
HateBR + ToLD-BR combinados.

## Arquitetura

```
[Streamlit]  --HTTP-->  [FastAPI]  -->  [Postgres]
                            |
                            +--> modelo.pkl (carregado uma vez, em memória)

[ml/treinar.py]  -->  lê Postgres + datasets  -->  escreve modelo.pkl
```

Princípio central: **`app/` nunca importa `ml/`**. A API só lê o artefato
treinado; quem treina é um script separado. Inferência e treino têm perfis de
recurso opostos (uma precisa de latência baixa e pouca RAM, o outro de muita RAM
por alguns minutos), e essa separação é o que permite hospedar a API em um plano
gratuito de 512 MB.

### Estrutura de arquivos

```
app/
  main.py                 FastAPI, lifespan carrega o modelo uma vez
  config.py               pydantic-settings: DATABASE_URL, JWT_SECRET, MODELO_PATH,
                          TREINO_HABILITADO
  database.py             engine, SessionLocal, get_db()
  models.py               SQLAlchemy: Usuario, Predicao, Exemplo, Treino
  schemas.py              Pydantic: entrada e saída de cada rota
  security.py             bcrypt + criação/validação de JWT
  ml.py                   carrega o .pkl, expõe prever(texto) -> (label, confianca)
  seed.py                 cria usuário admin, importa o CSV legado
  routers/
    auth.py  predicoes.py  exemplos.py  treinos.py  metricas.py

ml/
  dados.py                junta HateBR + ToLD-BR + tabela exemplo
  treinar.py              5-fold CV, só substitui o artefato se o F1 subir
  artefatos/modelo.pkl    pipeline sklearn completo, um arquivo só

frontend/app.py           Streamlit, apenas chamadas HTTP
tests/                    pytest + httpx, SQLite em memória
alembic/                  migrations
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
.github/dependabot.yml
.pre-commit-config.yaml
```

### Decisões de arquitetura

**Um artefato, não dois.** O `retreinar_modelo.py` atual monta um `Pipeline`
sklearn e então salva os dois passos separadamente, o que permite que modelo e
vetorizador saiam de sincronia. Passa a salvar o pipeline inteiro:
`joblib.dump(pipeline, ...)` e `pipeline.predict([texto])`.

**Modelo carregado no startup.** Hoje o carregamento acontece como efeito
colateral de `import testar_modelo`. Passa a ser explícito, no `lifespan` do
FastAPI, uma vez por processo.

**Retreino desligado em produção.** A flag `TREINO_HABILITADO` (default `false`
em produção, `true` em desenvolvimento) controla o `POST /treinos`. Com 512 MB de
RAM, `carregar_dados()` sobre ~28 mil linhas mais cross-validation não cabe. Em
produção a rota devolve `503` com mensagem explicativa. Isso é documentado no
README, não escondido.

## Modelo de dados

| Tabela | Colunas | Substitui |
|---|---|---|
| `usuario` | `id`, `email` (unique), `senha_hash`, `criado_em` | env var `HATEBR_SENHA_HASH` |
| `predicao` | `id`, `texto`, `label`, `confianca`, `criado_em` | — (dado novo) |
| `exemplo` | `id`, `texto` (unique), `label`, `usuario_id` (FK), `criado_em` | `feedback_treino.csv` |
| `treino` | `id`, `status`, `f1_macro`, `desvio`, `seed`, `qtd_exemplos`, `iniciado_em`, `terminado_em`, `erro` | `best_score.json` + `historico_execucoes.log` |

Detalhes:

- `exemplo.texto` é `UNIQUE`. Hoje o CSV aceita a mesma frase repetida, o que
  enviesa o treino. A constraint resolve na camada correta e a API devolve `409`.
- `treino.status` é um enum: `pendente`, `rodando`, `concluido`, `falhou`. É o que
  torna a resposta `202 Accepted` honesta — o cliente tem onde consultar o
  desfecho.
- Não existe endpoint de registro de usuário. Só o dono treina o modelo; usuários
  nascem pelo `app/seed.py`. Menos superfície de ataque, menos código.
- `predicao` não guarda IP nem qualquer identificador de quem digitou. Só o texto
  e o resultado. O README registra essa decisão citando LGPD.

### Fluxos

Predição:

```
Streamlit -> POST /predicoes {"texto": "..."}
          -> app.ml.prever()  (modelo já em memória)
          -> INSERT em predicao
          -> 201 {"label": 1, "rotulo": "Discurso de Ódio", "confianca": 0.87}
```

Retreino:

```
POST /treinos (JWT) -> INSERT treino (status=pendente) -> 202 {"id": 7}
                    -> BackgroundTasks:
                         status=rodando
                         ml/dados.py lê HateBR + ToLD-BR + tabela exemplo
                         cross_val_score 5-fold, f1_macro
                         substitui o artefato apenas se f1 > melhor f1 registrado
                         status=concluido (ou falhou, com erro preenchido)
GET /treinos/7 (JWT) -> status e métricas atuais
```

## Contrato da API

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

### Autenticação

- `bcrypt` diretamente, sem `passlib` (semi-abandonado). Salt embutido, custo
  configurável.
- `PyJWT`, HS256, expiração de 60 minutos.
- `OAuth2PasswordBearer`, que habilita o botão **Authorize** no Swagger — um
  recrutador consegue autenticar e testar rotas protegidas sem sair do browser.
- O `JWT_SECRET` vem de variável de ambiente, gerado pelo Render em produção.

Evolução em relação ao esquema atual, documentada no README: SHA-256 sem salt em
env var passa a bcrypt com salt no banco. O `hmac.compare_digest` permanece na
comparação de tokens.

### Validação nos limites de confiança

- `texto`: `Field(min_length=1, max_length=1000)`, com `strip` antes da validação.
- `label`: `Literal[0, 1]`.
- Rate limit de 30 requisições por minuto por IP no `POST /predicoes` (`slowapi`).
  É rota pública na internet aberta e inferência custa CPU.

### Erros

Um `exception_handler` global padroniza todas as respostas de erro no formato
`{"detail": "..."}`, incluindo os `422` gerados pelo Pydantic, para que o cliente
nunca precise lidar com dois formatos.

### Rotas de suporte

- `GET /health` verifica conexão com o banco e presença do modelo em memória.
  Não devolve `200` fixo — é o que o Render usa para decidir se o container subiu.
- `GET /metricas` devolve apenas agregados: total de predições, percentual
  classificado como ódio, total de exemplos ensinados, F1 do modelo em uso, data
  do último treino concluído.

## Testes

Desenvolvimento orientado a testes: teste primeiro, implementação depois, em
todas as fases.

- `conftest.py` com SQLite em memória e `dependency_overrides` sobre `get_db`.
  Testes não encostam no Postgres real.
- `app.ml.prever` substituído por um fake nos testes de rota. Teste de API não
  pode depender de um `.pkl` de 230 KB nem da acurácia do modelo.
- Casos cobertos: `401` sem token, `409` em texto duplicado, `422` em texto vazio
  e acima de 1000 caracteres, `201` no caminho feliz, `503` com modelo ausente,
  ciclo completo `202` seguido de `GET /treinos/{id}`.
- `ml/dados.py` testado com CSVs falsos em `tmp_path`, sem exigir o dataset real
  baixado.

## Automação

| O quê | Como | Quando dispara |
|---|---|---|
| Lint e formatação | `ruff` via `pre-commit` | a cada commit local |
| Testes e lint | GitHub Actions com service container Postgres | todo push e PR |
| Migrations | `alembic upgrade head` no start do container | todo deploy |
| Deploy | Render, auto-deploy ligado ao repositório | push na `main` que passa no CI |
| Atualização de dependências | `dependabot.yml` | semanalmente, abre PR sozinho |
| Proteção do modelo | `ml/treinar.py` só substitui o artefato se o F1 subir | a cada retreino |

Não haverá Makefile: `make` não existe por padrão no Windows, e o próprio
`docker compose` já serve de executor de tarefas.

```
docker compose up                                   sobe Postgres, API e Streamlit
docker compose run --rm api pytest                  testes
docker compose run --rm api alembic upgrade head    migrations
docker compose run --rm api python -m app.seed      cria usuário e importa o CSV legado
docker compose run --rm api python -m ml.treinar    retreina
```

### Senha do administrador

`app/seed.py --gerar-senha` gera a senha com `secrets.token_urlsafe(15)`
(aproximadamente 120 bits de entropia, via CSPRNG do sistema operacional),
exibe uma única vez no terminal e grava apenas o hash bcrypt no banco. A senha
não é escrita em arquivo, log ou histórico de shell. `--resetar-senha` gera outra
caso a original se perca.

Sem a flag, o script pede a senha via `getpass`, que não ecoa o que é digitado —
diferente do `input()` usado hoje no `gerar_hash_senha.py`.

## Distribuição do modelo treinado

O `modelo.pkl` está hoje no `.gitignore` e o deploy precisa dele dentro da
imagem. Decisão: versionar o artefato no repositório (cerca de 230 KB). É obra
derivada, não redistribui os datasets originais, e o README mantém os links e as
licenças de HateBR e ToLD-BR. Os datasets brutos continuam fora do repositório.

## Fases

Cada fase termina com o projeto funcionando e um commit.

1. **Dias 1-2** — FastAPI, `POST /predicoes`, `GET /health`, testes. Sem banco.
2. **Dias 3-4** — Postgres, SQLAlchemy, Alembic, gravação em `predicao`,
   `docker compose up` funcionando de ponta a ponta.
3. **Dias 5-6** — auth JWT, `/exemplos`, `app/seed.py`, importação do CSV legado.
4. **Dias 7-8** — `/treinos` em background, `/metricas`, separação de `ml/`.
5. **Dias 9-10** — CI, Dependabot, deploy no Render, link público no ar.
6. **Dias 11-12** — Streamlit consumindo a API, README reescrito com arquitetura,
   decisões tomadas e o histórico de evolução do projeto.

## Dependências do usuário

Três pontos exigem ação humana e não podem ser automatizados:

1. Repositório no GitHub.
2. Conta no Render e conexão OAuth do repositório.
3. Guardar a senha gerada pelo seed em um gerenciador de senhas.

## Fora de escopo

- Frontend em React ou Next.js. O Streamlit como cliente HTTP já demonstra a
  separação de camadas dentro do prazo de duas semanas.
- Fila de tarefas (Celery, Redis, RQ). `BackgroundTasks` resolve o caso de um
  único usuário; fila seria complexidade sem demanda.
- Troca do modelo por transformers (BERTimbau). O ganho seria de domínio, não de
  engenharia, e não cabe em 512 MB.
- Cadastro público de usuários, recuperação de senha, papéis e permissões.
- Observabilidade além de logs estruturados.
