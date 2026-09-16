import pandas as pd
import joblib
import random
import json
import os

from dados_treino import carregar_dados

seed = random.randint(0, 999_999)

dados = carregar_dados()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score 


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(class_weight="balanced"))
])

validador = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
scores = cross_val_score(pipeline, dados["comentario"], dados["label_final"], cv=validador, scoring="f1_macro")

f1_macro = scores.mean()
desvio = scores.std()

print (f"Scores por fold : {scores}")  
print (f"F1 macro médio : {f1_macro:.4f} (desvio: {desvio:.4f})")

if os.path.exists("best_score.json"):
    with open ("best_score.json", "r", encoding="utf-8") as f:
        melhor = json.load(f)
else:
    melhor = None

novo_melhor = melhor is None or f1_macro > melhor["f1_macro"] 

if novo_melhor:
    pipeline.fit(dados["comentario"], dados["label_final"])
    joblib.dump(pipeline.named_steps["clf"], "modelo_logistic_regression.pkl")
    joblib.dump(pipeline.named_steps["tfidf"], "vetorizador.pkl")
    with open("best_score.json", "w", encoding="utf-8") as f:
        json.dump({"f1_macro": f1_macro, "seed": seed}, f,)
