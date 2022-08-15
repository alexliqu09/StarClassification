from transformers import BertTokenizerFast, BertModel
from torch import nn
import torch


class BERTForClassification(nn.Module):
    def __init__(self, NAME_MODEL, is_Dropout=True):
        super(BERTForClassification, self).__init__()
        self.bert = BertModel.from_pretrained(NAME_MODEL)
        self.is_Dropout = is_Dropout
        self.do = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        if self.is_Dropout:
            cls_output = self.do(cls_output)
        output = self.linear(cls_output)
        return output


class BERTInference():
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

def getBERTModel(NAME_MODEL):
    tokenizer = BertTokenizerFast.from_pretrained(NAME_MODEL)
    model = BERTForClassification(NAME_MODEL, is_Dropout =False)
    return BERTInference(model, tokenizer, './weight/epoch_0_bert_18000_weightModelBERT.pth')
    

def getBERTInferenceModel(model, review):
    return model.inference_star(review)