from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Exemplo, Predicao
from app.schemas import Metricas

router = APIRouter(tags=["metricas"])


@router.get("/metricas", response_model=Metricas)
def obter_metricas(db: Session = Depends(get_db)) -> Metricas:
    total = db.scalar(select(func.count()).select_from(Predicao)) or 0
    odio = db.scalar(select(func.count()).select_from(Predicao).where(Predicao.label == 1)) or 0
    exemplos = db.scalar(select(func.count()).select_from(Exemplo)) or 0
    return Metricas(
        total_predicoes=total,
        taxa_odio=round(odio / total, 4) if total else 0.0,
        total_exemplos=exemplos,
        f1_modelo=None,
        ultimo_treino_em=None,
    )
