import pandas as pd


 
"""
This file checks if all the examples in the training set contain the given metaphor words in the exact form or a
different word form. 

If there are different word forms, then we might have to check for different word forms too. 

"""


data=pd.read_csv("train.csv")
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



data=pd.read_csv("../data/train.csv")

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

    data=add_target_index_and_target_word_to_data(data)
    return data

data=get_data()
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
texts,labels,target,target_index=get_data_for_melbert(data)
    
df = pd.DataFrame({
    'texts': texts,
    'labels': labels,
    'target': target,
    'target_index': target_index
})


# Save the DataFrame to a CSV file
df.to_csv('processed_data.csv', index=False)