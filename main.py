from data_loader import get_data,get_texts_labels,get_data_for_melbert,get_data_for_melbert_2
import pandas as pd
from train import base_bert_model,melbert_model
 
if __name__=="__main__":
    data=get_data()
    texts,labels=get_texts_labels(data)
    zero_cnt=0
    ones_cnt=0
    for i in range(len(labels)):
        if labels[i]==0:
            zero_cnt+=1
        elif labels[i]==1:
            ones_cnt+=1
    #print("Zero count : ",zero_cnt)
    #print("Ones count : ",ones_cnt)
    base_bert_model(texts,labels)

    #texts,labels,target,target_index=get_data_for_melbert(data)
    texts,labels,target,target_index=get_data_for_melbert_2(data)
    melbert_model(texts,labels,target,target_index)
    
    