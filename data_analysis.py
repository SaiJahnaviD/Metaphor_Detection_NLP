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

contains_target_word=0
target_word_not_found=0
total_rows,_=data.shape
for index,row in data.iterrows():
    met_id=row["metaphorID"]
    text=row["text"]
    text=text.lower()
    words=text.split()
    target_word=metaphor_id[met_id]
    plural_word=plurals[met_id]
    if(target_word in words):
        contains_target_word+=1
    else:
        if plural_word in words:
            contains_target_word+=1
        else :

            target_word_not_found+=1
            print(target_word)
            print(words)
            print()
            print()
print("Total rows : ",total_rows)
print("Rows containing target word : ",contains_target_word)
print("Rows not containing target word : ",target_word_not_found)



