import streamlit as st 
from utils import *

controller = Controller()

num_start = 0

if __name__ == '__main__':
    st.title("Clasificación de Estrellas")
    review = st.text_area('Ingrese una review', '')
    button = st.button('Predecir')

    opcion = st.sidebar.selectbox('Modelos', ("BERT", "DistillmBERT", "RoBERTa", "Electra"))
     
    if review != '' and button:
            if opcion == "BERT":
                num_start = controller.prediction(review, opcion)   
            
            elif opcion == "DistillmBERT":
                num_start = controller.prediction(review, opcion)   

            elif opcion == "RoBERTa":
                num_start = controller.prediction(review, opcion)

            elif opcion == "Electra":
                num_start = controller.prediction(review, opcion)   


            st.markdown(f"{num_start} estrellas")
            st.markdown(f"""
                <div>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                    {getStar(num_start)}
                </div>
            """,
                unsafe_allow_html=True
            )
