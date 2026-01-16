import pandas as pd

metaphor_id={
    0: "road",
1: "candle",
2: "light",
3: "spice",
4: "ride",
5: "train",
6: "boat"
}

 
plurals={
0: "roads",
1: "candles",
2: "lights",
3: "spices",
4: "rides",
5: "trains",
6: "boats"

}


def add_target_index_and_target_word_to_data(data):
    target=[]
    target_index=[]
    for index,row in data.iterrows():
        met_id=row["metaphorID"]
        text=row["text"]
        text=text.lower()
        target_word=metaphor_id[met_id]
        plural_word=plurals[met_id]
        words=text.split()
        if target_word in words:

            word_index=words.index(target_word)
            target.append(target_word)
            target_index.append(word_index)
        else:
            word_index=words.index(plural_word)
            target.append(plural_word)
            target_index.append(word_index)

    data["target"]=target
    data["target_index"]=target_index
    data['label'] = data['label_boolean'].apply(lambda x: 1 if x else 0)
    return data

    
def get_data():
    data=pd.read_csv("../data/train.csv")
    data = data.sample(frac=1, random_state=42)
    data=add_target_index_and_target_word_to_data(data)
    return data
    
def get_texts_labels(data):
    texts=data["text"].tolist()
    labels=data["label"].tolist()
    return texts,labels



def get_data_for_melbert(data):
    texts=data["text"].tolist()
    labels=data["label"].tolist()
    target=data["target"].tolist()
    target_index=data["target_index"].tolist()
    formatted_texts=[]
    for i in range(len(texts)):
        text=texts[i]
        indices = [i for i in range(len(text) - 2) if text[i:i+3] == " . "]
        required_index=0
        for j in range(len(indices)):
            if indices[j]>target_index[i]:
                required_index=indices[j]
                break
        formatted_texts.append(text[:required_index+2])
    return formatted_texts,labels,target,target_index

def get_data_for_melbert_2(data):
    texts=data["text"].tolist()
    labels=data["label"].tolist()
    target=data["target"].tolist()
    target_index=data["target_index"].tolist()
    print()
    print("Len of target_index : ",len(target_index))
    print("Len of texts : ",len(texts))
    formatted_texts=[]
    formatted_indices=[]
    formatted_labels=[]
    for i in range(len(texts)):
        text=texts[i]
        words=text.split()
        indices=[]
        #print(words)
        for j in range(len(words)):
          if words[j]==".":
            indices.append(j)

        required_index=0
        for j in range(len(indices)):
            if indices[j]>target_index[i]:
                required_index=indices[j]
                break
        if required_index!=0:
          start_ind=max(required_index-50,0)
          end_ind=max(required_index+50,len(words)-1)
          before_words=words[start_ind:required_index+1]
          after_words=words[required_index+1:end_ind]
          all_words=before_words+after_words
          sentence = ' '.join(all_words)
          formatted_texts.append(sentence)
          if max(required_index-50,0)==0:
            formatted_index=required_index
          else:
            formatted_index=50
          formatted_indices.append(formatted_index)
          formatted_labels.append(labels[i])
          
    return formatted_texts,formatted_labels,target,formatted_indices

    

