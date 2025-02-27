import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from utils.utils_data import data_collator
import logging

log = logging.getLogger()

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
    
class UeEstimatorTrustscore:
    def __init__(self, cls, config, train_dataset):
        self.cls = cls
        self.config = config
        self.train_dataloader = DataLoader(CustomDataset(train_dataset), batch_size=8, shuffle=False, collate_fn=data_collator)
    
    def __call__(self, X, y):
        test_hidden_states, test_answers = self._exctract_features_preds(X)
        eval_results = {"trust_score":[]}
        for hidden_state, answer in zip(test_hidden_states, test_answers):
            diffclass_dist = self._diffclass_euclid_dist(hidden_state, int(answer), self.train_hidden_states)
            sameclass_dist= self._sameclass_euclid_dist(hidden_state, int(answer), self.train_hidden_states)
            if sameclass_dist is None:
                eval_results["trust_score"].append(float(0.))
            else:
                if (diffclass_dist + sameclass_dist) == 0:
                    turst_score = 0.5
                else:
                    trust_score = diffclass_dist / (diffclass_dist + sameclass_dist)
                if np.isnan(trust_score):
                    print("=================nanananananann==================")
                    print("diff_class_dist:", diffclass_dist)
                    print("same_class_dist:", sameclass_dist)
                eval_results["trust_score"].append(float(trust_score))
        return eval_results

    def fit_ue(self, X=None, y=None, X_test=None):
        model = self.cls
        model = model.cuda()
        model.eval()

        log.info(
            "****************Start calcurating hiddenstate on train dataset **************"
        )
        hidden_states = []
        labels = []
        for step, inputs in enumerate(self.train_dataloader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
            labels.append(inputs["labels"].to('cpu').detach().numpy().copy())
        hidden_states = np.concatenate(hidden_states)
        labels = np.concatenate(labels)
        labels_hidden_states = defaultdict(list)
        for label, hidden_state in zip(labels, hidden_states):
            labels_hidden_states[int(label)].append(hidden_state)
        self.train_hidden_states = labels_hidden_states
        log.info("**************Done.**********************")

    def _exctract_features_preds(self, X):
        model = self.cls
        model = model.cuda()
        test_dataloader = DataLoader(CustomDataset(X), batch_size=8, shuffle=False, collate_fn=data_collator)
    
        hidden_states = []
        answers = []
        for step, inputs in enumerate(test_dataloader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
            answers.append(np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=-1))
            
        hidden_states = np.concatenate(hidden_states)
        answers = np.concatenate(answers)
        return hidden_states, answers
    
    def _diffclass_euclid_dist(self, test_hidden_state, test_answer, train_hiddens_labels):
        min_dist = None
        for train_label, train_hidden_states in train_hiddens_labels.items():
            if int(train_label) != int(test_answer):
                for train_hidden_state in train_hidden_states:
                    dist = np.linalg.norm(test_hidden_state-train_hidden_state)
                    if(min_dist is None or dist < min_dist):
                        min_dist = dist
        return min_dist
    
    def _sameclass_euclid_dist(self, test_hidden_state, test_answer, train_hiddens_labels):
        min_dist = None
        for train_label, train_hidden_states in train_hiddens_labels.items():
            if int(train_label) == int(test_answer):
                for train_hidden_state in train_hidden_states:
                    dist = np.linalg.norm(test_hidden_state-train_hidden_state)
                    if(min_dist is None or dist < min_dist):
                        min_dist = dist
        return min_dist