import testar_modelo
import streamlit as st

modelo = testar_modelo.modelo
vetorizador = testar_modelo.vetorizador



st.title("Identificação de Comentários de Ódio")
st.header("Digite um comentário para verificar se é considerado discurso de ódio ou não.")


def verificar_comentario(comentario):
    if comentario:
        comentario_vetorizado = vetorizador.transform([comentario])
        predicao = modelo.predict(comentario_vetorizado)[0]
        resultado = "Discurso de Ódio" if predicao == 1 else "Não é Discurso de Ódio"
        return resultado

st.text_input("Digite seu comentário aqui:", key="comentario")
clicada = st.button("Verificar")

if clicada:
    comentario = st.session_state.comentario
    resultado = verificar_comentario(comentario)
    st.write(f"Resultado: {resultado}")

# C:\Users\ruper\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run classificador.py (CÓDIGO DE EXECUÇÃO DO STREAMLIT)

