from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.utils_data import data_collator

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


def evaluate_model(config, model, datasets):
    test_dataset = CustomDataset(datasets['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    
    model.eval()
    hidden_states = []
    reg_output = []
    logits = []
    for step, inputs in enumerate(test_dataloader):
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        if config.model.model_type == "hybrid":
            reg_output.append(outputs.reg_output.to('cpu').detach().numpy().copy())
            logits.append(outputs.logits.to('cpu').detach().numpy().copy())
    reg_output = list(np.concatenate(reg_output))
    logits = list(np.concatenate(logits))
    hidden_states = list(np.concatenate(hidden_states))

    return {"reg_output":reg_output, "logits":logits, "hidden_states":hidden_states}