import streamlit as st 
from utils import *

num_start = 0

STYLE='align="justify"'
SUMMARIES = {"BERT": f"""
                <p {STYLE}>
                    "The Spanish language is one of the top 5 spoken languages in the world. Nevertheless, 
                     finding resources to train or evaluate Spanish language models is not an easy task. 
                     In this paper we help bridge this gap by presenting a BERT-based language model pre-trained 
                     exclusively on Spanish data. As a second contribution, we also compiled several tasks 
                     specifically for the Spanish language in a single repository much in the spirit of the 
                     GLUE benchmark. By fine-tuning our pre-trained Spanish model we obtain better results 
                     compared to other BERT-based models pre-trained on multilingual corpora for most of the 
                     tasks, even achieving a new state-of-the-art on some of them. We have publicly released our 
                     model, the pre-training data and the compilation of the Spanish benchmarks."
                </p>
                Extraido de: 
                José Cañete et al. “Spanish Pre-Trained BERT Modeland Evaluation Data”. 
                
                Paper original <a href="https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf">aquí</a>
            """,
            
            "DistillmBERT": f"""
                <p {STYLE}>
                    "As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural
                    Language Processing (NLP), operating these large models in on-the-edge and/or under constrained
                    computational training or inference budgets remains challenging. In this work, we propose a
                    method to pre-train a smaller general-purpose language representation model, called DistilBERT,
                    which can then be fine-tuned with good performances on a wide range of tasks like its larger
                    counterparts. While most prior work investigated the use of distillation for building task-specific
                    models, we leverage knowledge distillation during the pre-training phase and show that it is
                    possible to reduce the size of a BERT model by 40%, while retaining 97\% of its language understanding
                    capabilities and being 60% faster. To leverage the inductive biases learned by larger models
                    during pre-training, we introduce a triple loss combining language modeling, distillation and
                    cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
                    demonstrate its capabilities for on-device computations in a proof-of-concept experiment and
                    a comparative on-device study."
                </p>
                Extraido de: 
                Victor Sanh et al. “Spanish Pre-Trained BERT Modeland Evaluation Data”. 
                
                Paper original <a href="https://arxiv.org/pdf/1910.01108.pdf">aquí</a>
            """,

            "RoBERTa": f"""
                <p {STYLE}>
                    "Language model pretraining has led to significant performance gains but careful comparison between
                    different approaches is challenging. Training is computationally expensive, often done on private
                    datasets of different sizes, and, as we will show, hyperparameter choices have significant impact
                    on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that
                    carefully measures the impact of many key hyperparameters and training data size. We find that BERT
                    was significantly undertrained, and can match or exceed the performance of every model published after
                    it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight
                    the importance of previously overlooked design choices, and raise questions about the source of
                    recently reported improvements. We release our models and code."
                </p>
                Extraido de: 
                Yinhan Liu et al. “Spanish Pre-Trained BERT Modeland Evaluation Data”. 
                
                Paper original <a href="https://arxiv.org/pdf/1907.11692.pdf">aquí</a>
            """,

            "Electra": f"""
                <p {STYLE}>
                    "Masked language modeling (MLM) pre-training methods such as BERT corrupt the input by replacing some
                    tokens with [MASK] and then train a model to reconstruct the original tokens. While they produce good
                    results when transferred to downstream NLP tasks, they generally require large amounts of compute to
                    be effective. As an alternative, we propose a more sample-efficient pre-training task called replaced
                    token detection. Instead of masking the input, our approach corrupts it by replacing some tokens with
                    plausible alternatives sampled from a small generator network. Then, instead of training a model that
                    predicts the original identities of the corrupted tokens, we train a discriminative model that predicts
                    whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments
                    demonstrate this new pre-training task is more efficient than MLM because the task is defined over all
                    input tokens rather than just the small subset that was masked out. As a result, the contextual representations
                    learned by our approach substantially outperform the ones learned by BERT given the same model size,
                    data, and compute. The gains are particularly strong for small models; for example, we train a model
                    on one GPU for 4 days that outperforms GPT (trained using 30x more compute) on the GLUE natural language
                    understanding benchmark. Our approach also works well at scale, where it performs comparably to RoBERTa
                    and XLNet while using less than 1/4 of their compute and outperforms them when using the same amount of
                    compute."
                </p>
                Extraido de: 
                Kevin Clark et al. “Spanish Pre-Trained BERT Modeland Evaluation Data”. 
                
                Paper original <a href="https://arxiv.org/pdf/2003.10555.pdf">aquí</a>
            """
}

if __name__ == '__main__':
    
    opcion = st.sidebar.selectbox('Modelos', ("Presentación", "BERT", "DistillmBERT", "RoBERTa", "Electra"))

    if opcion == 'Presentación':
        st.title("Clasificación de Estrellas")
        st.markdown(f"""
            <h2><b>Sobre nuestro trabajo</b></h2>
            <div {STYLE}>
            <p>
                El campo de la mineria de texto ha tenido una gran acogida en los ultimos años debido al interes
                de las empresas por conocer la apreciación y perseccion de sus productos por el lado de los clientes,
                por lo que el análisis de las opiniones de las compradores es de vital importancia, agregando que también
                es importante para los usuarios al decidir que producto comprar y/o recomendar, por lo que una manera
                entendible de poder evaluar sus opiniones es en base a la calificación de estrellas.
            </p>
            
            <p>
                Teniendo presente los avances del Deep Learning en tareas como el análisis de sentimiento, detección
                de emociones, entre otros, en este trabajo nos enfocamos en la tarea de predicción de estrellas usando
                modelos de Deep Learning.\n
            </p>
            
            <p>
                Se realizó una comparativa entre los modelos transformers más populares, donde todos los modelos fueron
                entrenados usando la técnica de fine tuning bajo el dataset de YELP y se obtuvo que el modelo con mejor
                resultado fue Beto bajo las métricas F1-score weight y Accuracy.
            </p>
            </div>
            <div>Nuestro trabajo se encuentra disponible <a href="https://drive.google.com/file/d/13boESDzS1ewTeojNOAnzW3fN692waVyb/view">aquí</a></div>
            <br/>
            <br/>
            <br/>
            """,
                unsafe_allow_html=True
            )

        st.image('src/pipeline_yelp.png', 'Pipiline del trabajo')
    else:
        st.title(f"Clasificación de Estrellas ({opcion})")
        st.markdown(SUMMARIES[opcion], unsafe_allow_html=True)

        review = st.text_area('Ingrese una review', '')
        button = st.button('Predecir')

        controller = Controller()
        

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
