import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
 
class BaseBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.1):
        super(BaseBERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
        
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        logits = self.linear(dropped_output)
        probabilities = self.softmax(logits)
        

        #return logits
        return probabilities



class MelBERTCLassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.1):
        super(MelBERTCLassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.SPV_layer = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
        self.MIP_layer = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask,input_ids_2,attention_mask_2,target_idx):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sentence_hidden_states=outputs.last_hidden_state
        word_representation_from_sentence = sentence_hidden_states[0][target_idx]
        sentence_output = outputs.pooler_output

        target_word=self.bert(input_ids_2, attention_mask=attention_mask_2)
        target_word_representation = target_word.pooler_output

        word_representation_from_sentence=self.dropout(word_representation_from_sentence)
        sentence_output = self.dropout(sentence_output)

        target_word_representation=self.dropout(target_word_representation)

        

        SPV_hidden = self.SPV_layer(torch.cat([sentence_output, word_representation_from_sentence], dim=1))
        MIP_hidden = self.MIP_layer(torch.cat([target_word_representation, word_representation_from_sentence], dim=1))
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        probabilities = self.softmax(logits)

        return probabilities
        









