from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from utils.utils_data import data_collator
from utils.score_range import upper_score_dic, asap_ranges

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
    model = model.cuda()
    model.eval()
    hidden_states = []
    reg_output = []
    logits = []
    for step, inputs in enumerate(test_dataloader):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        if config.model.model_type == "hybrid":
            reg_output.append(outputs.reg_output.to('cpu').detach().numpy().copy())
            logits.append(outputs.logits.to('cpu').detach().numpy().copy())
    reg_output = np.concatenate(reg_output).tolist()
    logits = np.concatenate(logits).tolist()
    hidden_states = np.concatenate(hidden_states).tolist()

    return {"reg_output":reg_output, "logits":logits, "hidden_states":hidden_states}

def calc_rpp(conf, squared_error):
  n = len(conf)
  cr_pair = list(zip(conf, squared_error))
  cr_pair.sort(key=lambda x: x[0], reverse=False)

  cnt = 0
  for i in range(n):
    for j in range(i, n):
      if(cr_pair[i][1] < cr_pair[j][1]):
        cnt += 1
  return cnt / (n**2)

def calc_predscore_conf(config, eval_results):
    if config.data.task_name == 'riken':
        high = upper_score_dic[config.data.question_id_suff][config.data.score_id]
        low = 0
    elif config.data.task_name == 'asap':
        low, high = asap_ranges[config.data.prompt_id]

    if config.model.model_type == 'hybrid':
        probs = softmax(eval_results['logits'], axis=1)
        pred_scores = np.round(np.array(eval_results['reg_output']).reshape(-1) * (high - low) + low)
        conf = [ps[int(i)] for ps, i in zip(probs, pred_scores)]
        return pred_scores, np.array(conf)

def eval_metric(config, eval_results, metric):
    if config.data.task_name == 'riken':
        high = upper_score_dic[config.data.question_id_suff][config.data.score_id]
        low = 0
    elif config.data.task_name == 'asap':
        low, high = asap_ranges[config.data.prompt_id]

    pred_scores, conf = calc_predscore_conf(config, eval_results)

    if metric == 'qwk':
        return cohen_kappa_score(np.array(eval_results['true_labels']) + low, pred_scores+low, labels = list(range(low, high + 1)), weights='quadratic')
    elif metric == 'corr':
        return np.corrcoef(np.array(eval_results['true_labels']) + low, pred_scores+low)[0][1]
    elif metric == 'roc':
        errors = (np.array(eval_results['true_labels']) != pred_scores).astype('int32')
        return roc_auc_score(errors, -conf)
    elif metric == 'rpp':
        squared_error = ((np.array(eval_results['true_labels']) + low) - (pred_scores+low)) ** 2
        return calc_rpp(conf=conf, squared_error=squared_error)