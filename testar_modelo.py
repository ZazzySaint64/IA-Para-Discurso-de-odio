import joblib


modelo = joblib.load('modelo_logistic_regression.pkl')
vetorizador = joblib.load('vetorizador.pkl')

X_teste_vetorizado = vetorizador.transform(['Comentário de teste para verificar se o modelo funciona corretamente.'])
previsoes = modelo.predict(X_teste_vetorizado)

print(previsoes)