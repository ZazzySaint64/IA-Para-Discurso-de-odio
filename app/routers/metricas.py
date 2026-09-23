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
    ultimo_treino_em = db.scalar(
        select(Treino.terminado_em)
        .where(Treino.status == "concluido")
        .order_by(Treino.terminado_em.desc())
        .limit(1)
    )
    # O melhor F1, não o do último treino: `ml/treinar.py` só troca o artefato
    # quando o score melhora, então o modelo em uso é sempre o do maior F1.
    f1_modelo = db.scalar(select(func.max(Treino.f1_macro)).where(Treino.status == "concluido"))
    return Metricas(
        total_predicoes=total,
        taxa_odio=round(odio / total, 4) if total else 0.0,
        total_exemplos=exemplos,
        f1_modelo=f1_modelo,
        ultimo_treino_em=ultimo_treino_em,
    )
