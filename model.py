from torch import nn
import torch.nn.functional as F
import torch
import sys 

sys.path.append('../')
sys.path.append('../../')
from transformers import T5ForConditionalGeneration


class CodeT5HeadWithValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.model_dim, 1)
        
    def load_base_model(self, load_model_path):
        self.model.load_state_dict(torch.load(load_model_path))

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, 
                                         decoder_attention_mask=decoder_attention_mask, output_hidden_states=True)
        hidden_states = outputs.decoder_hidden_states[-1]
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, outputs, value)
        return outputs

    
def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=100, top_p=1.0):
    
    preds = model.model.generate(source_ids, attention_mask=attention_mask, do_sample=True, top_k=top_k, top_p=top_p,
                                 max_length=max_target_length)
    # preds = model.module.model.generate(source_ids, attention_mask=attention_mask, do_sample=True, top_k=top_k, top_p=top_p,
    #                              max_length=max_target_length)
    return preds