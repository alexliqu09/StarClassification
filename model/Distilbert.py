from transformers import  DistilBertModel, DistilBertTokenizerFast
import torch.nn as nn
import torch 

class DistillmBERTModel(nn.Module):
    def __init__(self, NAME_MODEL, is_Dropout=True):
        super(DistillmBERTModel, self).__init__()
        self.num_labels = 5
        self.distilbert = DistilBertModel.from_pretrained(NAME_MODEL)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.is_Dropout = is_Dropout
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids=None, attention_mask=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask)
        hidden_state = distilbert_output[0]                    
        pooled_output = hidden_state[:, 0]                   
        if self.is_Dropout:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 
        return logits


class DistllmBERTInference():
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
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.MAX_LEN:
            tokens = tokens + ['[PAD]' for _ in range(self.MAX_LEN-len(tokens))]
        elif len(tokens) > self.MAX_LEN:
            tokens = tokens[:self.MAX_LEN-1] + ['[SEP]']  
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(token_ids).flatten().to(self.device)
        input_ids = torch.unsqueeze(tokens_ids_tensor, dim=0)
        attn_mask = (tokens_ids_tensor != 0).long().flatten().to(self.device)
        attention_mask = torch.unsqueeze(attn_mask, dim=0)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output,dim=1)
        return prediction.item()+1


def getDistilmBERTModel(NAME_MODEL):
    tokenizer = DistilBertTokenizerFast.from_pretrained(NAME_MODEL)
    model = DistillmBERTModel(NAME_MODEL, is_Dropout =False)
    return DistllmBERTInference(model, tokenizer, './weight/epoch_0_distilbert_18000.pth')
    

def getDistilmBERTInferenceModel(model, review):
    return model.inference_star(review)