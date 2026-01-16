import torch
 
class MetaphorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MelBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels,target,target_index,sentences):
        self.encodings = encodings
        self.sentences=sentences
        self.labels = labels
        self.target_encodings = target
        self.target_index = target_index

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item.update({key+'_2': torch.tensor(val[idx]) for key, val in self.target_encodings.items()})
        item["sentence"]=self.sentences[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        item['target_index'] = torch.tensor(self.target_index[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
