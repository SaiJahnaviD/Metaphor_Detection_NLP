from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from metaphordataset import MetaphorDataset, MelBERTDataset
from models import BaseBERTClassifier, MelBERTCLassifier
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
 

def melbert_test(texts,labels,target,target_index):
    batch_size=8
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  
    model = MelBERTCLassifier(num_classes=num_classes)
    model.load_state_dict(torch.load("saved_model.pth"))
    model.eval()

    data = list(zip(texts, labels, target,target_index))

    test_texts, test_labels, test_target, test_target_index = zip(*data)
    
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_target_encodings=tokenizer(test_target, truncation=True, padding=True)
    

    test_dataset = MelBERTDataset(test_encodings, test_labels,test_target_encodings, test_target_index,test_texts)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    all_preds=[]
    all_labels=[]
    with torch.no_grad():
            
        for batch in test_loader:
            input_ids_1= batch['input_ids'].to(device)
            attention_mask_1= batch['attention_mask'].to(device)
            input_ids_2= batch['input_ids_2'].to(device)
            attention_mask_2= batch['attention_mask_2'].to(device)
        
            labels = batch['labels'].to(device)
            target_index=batch["target_index"].to(device)
        
            outputs = model(input_ids_1, attention_mask_1,input_ids_2,attention_mask_2,target_index)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
                
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precison = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print("MelBERT Model : ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision Score: {precison:.4f}")
    print(f"Recall Score: {recall:.4f}")


