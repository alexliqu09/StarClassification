import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast


NAME_MODEL = "mrm8488/electricidad-base-discriminator"

class ElectraInference():
    def __init__(self, NAME_MODEL, weight):
        self.model = ElectraForSequenceClassification.from_pretrained(NAME_MODEL, num_labels = 5)
        self.tokenizer = ElectraTokenizerFast.from_pretrained('mrm8488/electricidad-base-discriminator')
        self.device = "cpu"
        self.model.to(self.device)
        self.load_weight = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(self.load_weight['state_dict'])
        self.MAXLEN = 128
        
    def inference_star(self, review):
        self.model.eval()
        tokens = self.tokenizer.tokenize(review) 
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.MAXLEN:
            tokens = tokens + ['[PAD]' for _ in range(self.MAXLEN-len(tokens))]
        elif len(tokens) > self.MAXLEN:
                tokens = tokens[:self.MAXLEN-1] + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(token_ids).flatten().to(self.device)
        input_ids = torch.unsqueeze(tokens_ids_tensor, dim=0)
        attn_mask = (tokens_ids_tensor != 1).long().flatten().to(self.device)
        attention_mask = torch.unsqueeze(attn_mask, dim=0).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output.logits,dim=1)
        return prediction.item()+1

def getElectraModel():
    return ElectraInference(NAME_MODEL, "./weight/Electra_0_16000.pth")