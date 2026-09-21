from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, get_db
from app.models import Treino, Usuario
from app.schemas import TreinoCriado, TreinoSaida
from app.security import usuario_atual

router = APIRouter(prefix="/treinos", tags=["treinos"])


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

        melhor = db.scalar(
            select(Treino.f1_macro)
            .where(Treino.status == "concluido")
            .order_by(Treino.f1_macro.desc())
            .limit(1)
        )
        resultado = treinar(db=db, f1_atual=melhor)

        treino.status = "concluido"
        treino.f1_macro = resultado["f1_macro"]
        treino.desvio = resultado["desvio"]
        treino.seed = resultado["seed"]
        treino.qtd_exemplos = resultado["qtd_exemplos"]
        treino.terminado_em = datetime.now(UTC)
        db.commit()
    except Exception as exc:  # noqa: BLE001 - o erro precisa virar registro, não sumir
        # Se o erro veio de um flush/commit (ex.: leitura dos exemplos), a sessão
        # fica suja e nem o commit abaixo funcionaria sem isto primeiro.
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
                "Retreino desligado em produção: o plano gratuito tem 512 MB de RAM, "
                "insuficiente para validação cruzada. Rode localmente com "
                "`python -m ml.treinar`."
            ),
        )

    rodando = db.scalar(select(Treino).where(Treino.status.in_(("pendente", "rodando"))))
    if rodando is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Já existe um treino em andamento (id {rodando.id})",
        )

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
