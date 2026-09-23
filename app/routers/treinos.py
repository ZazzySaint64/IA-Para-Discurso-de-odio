import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import ml
from app.config import settings
from app.database import SessionLocal, get_db
from app.models import Treino, Usuario
from app.schemas import TreinoCriado, TreinoSaida
from app.security import usuario_atual

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/treinos", tags=["treinos"])

# Treino que não terminou em uma hora foi morto junto com o processo: nenhum
# except roda quando o processo leva SIGKILL. Sem isso, um OOM trava todo
# treino futuro no 409.
LIMITE_TREINO_TRAVADO = timedelta(hours=1)


def _em_utc(momento: datetime) -> datetime:
    """Normaliza para um datetime ciente de fuso, em UTC.

    A coluna é DateTime(timezone=True), mas o SQLite não guarda fuso de
    verdade: o valor volta sem tzinfo (Postgres volta com). Sem isto, comparar
    com datetime.now(UTC) levanta TypeError só no SQLite (visto nos testes),
    então a normalização tem que rodar em Python, não no WHERE do SQL.
    """
    return momento if momento.tzinfo is not None else momento.replace(tzinfo=UTC)


def _executar_treino(treino_id: int, db_factory=SessionLocal) -> None:
    """Roda em background, com sessão própria.

    A sessão do request já foi fechada quando isto executa, por isso abre outra.
    O import de ml.treinar fica aqui dentro de propósito: assim pandas e sklearn
    não entram no caminho de import da API.
    """
    from ml.treinar import treinar

    db = db_factory()
    try:
        treino = db.get(Treino, treino_id)
        treino.status = "rodando"
        db.commit()
        logger.info("Treino %s iniciado", treino_id)

        melhor = db.scalar(
            select(Treino.f1_macro)
            .where(Treino.status == "concluido")
            .order_by(Treino.f1_macro.desc())
            .limit(1)
        )
        resultado = treinar(db=db, f1_atual=melhor)
        logger.info(
            "Treino %s terminou: f1_macro=%.4f, artefato %s",
            treino_id,
            resultado["f1_macro"],
            "substituído" if resultado["substituiu"] else "mantido",
        )

        # Sem isto a API segue servindo o pipeline carregado no lifespan, ou
        # seja, o modelo antigo, até o processo reiniciar. Religar a global do
        # módulo é atômico no CPython: não precisa de lock.
        if resultado["substituiu"]:
            ml.carregar_modelo(Path(settings.MODELO_PATH))

        treino.status = "concluido"
        treino.f1_macro = resultado["f1_macro"]
        treino.desvio = resultado["desvio"]
        treino.seed = resultado["seed"]
        treino.qtd_exemplos = resultado["qtd_exemplos"]
        treino.substituiu = resultado["substituiu"]
        treino.terminado_em = datetime.now(UTC)
        db.commit()
    except Exception as exc:  # noqa: BLE001 - o erro precisa virar registro, não sumir
        # Se o erro veio de um flush/commit (ex.: leitura dos exemplos), a sessão
        # fica suja e nem o commit abaixo funcionaria sem isto primeiro.
        logger.error("Treino %s falhou: %s", treino_id, exc)
        db.rollback()
        treino = db.get(Treino, treino_id)
        treino.status = "falhou"
        treino.erro = str(exc)[:500]
        treino.terminado_em = datetime.now(UTC)
        db.commit()
    finally:
        db.close()


@router.post("", response_model=TreinoCriado, status_code=status.HTTP_202_ACCEPTED)
def criar_treino(
    tarefas: BackgroundTasks,
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> TreinoCriado:
    if not settings.TREINO_HABILITADO:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retreino desligado nesta API: em produção o plano gratuito tem 512 MB "
                "de RAM, insuficiente para validação cruzada; localmente (docker "
                "compose) a imagem não leva os datasets, e o treino roda fora do "
                "processo da API por design. Rode com `python -m ml.treinar`."
            ),
        )

    rodando = db.scalar(select(Treino).where(Treino.status.in_(("pendente", "rodando"))))
    if rodando is not None:
        idade = datetime.now(UTC) - _em_utc(rodando.iniciado_em)
        if idade <= LIMITE_TREINO_TRAVADO:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Já existe um treino em andamento (id {rodando.id})",
            )
        rodando.status = "falhou"
        rodando.erro = (
            "Treino abandonado: passou de 1h sem terminar, processo "
            "provavelmente morreu (ex.: OOM) antes de gravar o desfecho."
        )
        rodando.terminado_em = datetime.now(UTC)
        db.commit()

    treino = Treino(status="pendente")
    db.add(treino)
    db.commit()
    tarefas.add_task(_executar_treino, treino.id, SessionLocal)
    return TreinoCriado(id=treino.id, status=treino.status)


@router.get("/{treino_id}", response_model=TreinoSaida)
def obter_treino(
    treino_id: int,
    db: Session = Depends(get_db),
    usuario: Usuario = Depends(usuario_atual),
) -> Treino:
    treino = db.get(Treino, treino_id)
    if treino is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Treino não encontrado")
    return treino
