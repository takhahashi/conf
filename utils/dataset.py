from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': item['input_ids'],
            'token_type_ids': item['token_type_ids'],
            'attention_mask': item['attention_mask'],
            'label': item['label']
        }