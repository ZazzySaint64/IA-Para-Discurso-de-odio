"""A API não pode depender de `ml/` em tempo de import.

Regex não serve aqui: foi um regex que deixou passar o `from ml.treinar import
treinar` adiado dentro de `_executar_treino`, e a documentação ficou afirmando
mais do que o código garante. O `ast` sabe a diferença entre um import no topo
do módulo (col_offset 0) e um dentro de função.
"""

import ast
from pathlib import Path

RAIZ_APP = Path(__file__).resolve().parent.parent / "app"


def _imports_de_ml_no_topo(arquivo: Path) -> list[str]:
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    achados = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Import | ast.ImportFrom) or no.col_offset != 0:
            continue
        if isinstance(no, ast.Import):
            achados += [a.name for a in no.names if a.name == "ml" or a.name.startswith("ml.")]
        elif no.level == 0 and no.module and (no.module == "ml" or no.module.startswith("ml.")):
            achados.append(no.module)
    return achados


def test_app_nao_importa_ml_no_nivel_de_modulo():
    arquivos = sorted(RAIZ_APP.rglob("*.py"))
    assert arquivos, "nenhum .py encontrado em app/ — o teste estaria passando à toa"
    ofensores = {str(a.relative_to(RAIZ_APP)): _imports_de_ml_no_topo(a) for a in arquivos}
    ofensores = {k: v for k, v in ofensores.items() if v}
    assert not ofensores, f"import de ml/ no topo do módulo: {ofensores}"
