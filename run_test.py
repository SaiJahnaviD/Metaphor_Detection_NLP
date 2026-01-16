from data_loader import get_data,get_texts_labels,get_data_for_melbert,get_data_for_melbert_2,add_target_index_and_target_word_to_data
import pandas as pd
from test import melbert_test



if __name__=="__main__":
    data=pd.read_csv("../data/test.csv")
    data = data.sample(frac=1, random_state=42)
    data=add_target_index_and_target_word_to_data(data)
    texts,labels,target,target_index=get_data_for_melbert_2(data)
    melbert_test(texts,labels,target,target_index)
 