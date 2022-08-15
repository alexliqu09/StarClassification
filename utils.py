from model.BERT import *
from model.Distilbert import *
from model.Electra import *
from model.RoBERTa import *

def getStar(num_start):
    star_style="""
        color:orange;
        font-size:2em;
    """

    text = ""
    for _ in range(num_start):
        text+=f'<span style="{star_style}" class="fa fa-star checked"></span>'        
    
    if num_start<5:
        star_style="""
            font-size:2em;
        """
        rest_num_star = 5 - num_start
        for _ in range(rest_num_star):
            text+=f'<span style="{star_style}" class="fa fa-star"></span>'

    return text

def Singleton(cls):
    istances = dict()
    def wrap(*args, **kwargs):
        if cls not in istances:
            istances[cls] = cls(*args, **kwargs)
        return istances[cls]

    return wrap

@Singleton
class Controller:
    def __init__(self):
        self.beto = getBERTModel("dccuchile/bert-base-spanish-wwm-cased")
        self.distilbert = getDistilmBERTModel("Geotrend/distilbert-base-es-cased")
        self.elctra = getElectraModel()
        self.roberta = getRoBERTaModel("bertin-project/bertin-roberta-base-spanish")

    def prediction(self, review, opt):
        if opt == "BERT":
            return getBERTInferenceModel(self.beto, review)

        elif opt == "DistillmBERT":
            return getDistilmBERTInferenceModel(self.distilbert, review)

        elif opt == "Electra":
            return self.elctra.inference_star(review)

        elif opt == "RoBERTa":
            return getRoBERTaInferenceModel(self.roberta, review)
        else:
            return "Model not Available"
