from transformers import RobertaModel, RobertaTokenizerFast
import torch


class RoBERTaForClassfication(torch.nn.Module):
    def __init__(self, NAME_MODEL, is_Dropout=True):
        super(RoBERTaForClassfication, self).__init__()
        self.roberta = RobertaModel.from_pretrained(NAME_MODEL, num_labels = 5, add_pooling_layer = False) 
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 5)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.is_Dropout = is_Dropout

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        if self.is_Dropout:
            pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class RoBERTaInference():
    def __init__(self, model, tokenizer, weight):
        self.device = "cpu"
        self.MAX_LEN = 128
        self.model = model
        self.model = self.model.to(self.device)
        self.tokenizer = tokenizer
        self.load_weight = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(self.load_weight['weight_model'])

    def inference_star(self, review):
        self.model.eval()  
        tokens = self.tokenizer.tokenize(review) 
        tokens = ['<s>'] + tokens + ['</s>']
        if len(tokens) < self.MAX_LEN:
            tokens = tokens + ['<pad>' for _ in range(self.MAX_LEN-len(tokens))]
        elif len(tokens) > self.MAX_LEN:
            tokens = tokens[:self.MAX_LEN-1] + ['</s>']  
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(token_ids).flatten().to(self.device)
        token_type_ids = torch.zeros_like(tokens_ids_tensor)
        token_type_ids = torch.unsqueeze(token_type_ids, dim=0)
        input_ids = torch.unsqueeze(tokens_ids_tensor, dim=0)
        attn_mask = (tokens_ids_tensor != 0).long().flatten().to(self.device)
        attention_mask = torch.unsqueeze(attn_mask, dim=0)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            _, prediction = torch.max(output,dim=1)
        return prediction.item()+1

def getRoBERTaModel(NAME_MODEL):
    tokenizer = RobertaTokenizerFast.from_pretrained(NAME_MODEL)
    model = RoBERTaForClassfication(NAME_MODEL, is_Dropout =False)
    return RoBERTaInference(model, tokenizer, './weight/Roberta_16000_weightModelBERT.pth')
    

def getRoBERTaInferenceModel(model, review):
    return model.inference_star(review)