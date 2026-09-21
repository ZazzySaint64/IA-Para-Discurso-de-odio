from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Exemplo, Predicao, Treino
from app.schemas import Metricas

router = APIRouter(tags=["metricas"])


@router.get("/metricas", response_model=Metricas)
def obter_metricas(db: Session = Depends(get_db)) -> Metricas:
    total = db.scalar(select(func.count()).select_from(Predicao)) or 0
    odio = db.scalar(select(func.count()).select_from(Predicao).where(Predicao.label == 1)) or 0
    exemplos = db.scalar(select(func.count()).select_from(Exemplo)) or 0
    ultimo = db.scalar(
        select(Treino)
        .where(Treino.status == "concluido")
        .order_by(Treino.terminado_em.desc())
        .limit(1)
    )
    return Metricas(
        total_predicoes=total,
        taxa_odio=round(odio / total, 4) if total else 0.0,
        total_exemplos=exemplos,
        f1_modelo=ultimo.f1_macro if ultimo else None,
        ultimo_treino_em=ultimo.terminado_em if ultimo else None,
    )
