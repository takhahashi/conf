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
    if config.model.model_type != "ensemble":
        model = model.cuda()
        model.eval()
        hidden_states = []
        reg_score = []
        reg_lnvar = []
        logits = []
        for step, inputs in enumerate(test_dataloader):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
            if config.model.model_type == "hybrid":
                reg_score.append(outputs.reg_score.to('cpu').detach().numpy().copy())
                logits.append(outputs.logits.to('cpu').detach().numpy().copy())
            elif config.model.model_type == "classification":
                logits.append(outputs.logits.to('cpu').detach().numpy().copy())
            elif config.model.model_type == "gaussianregression":
                reg_score.append(outputs.pred_score.to('cpu').detach().numpy().copy())
                reg_lnvar.append(outputs.pred_lnvar.to('cpu').detach().numpy().copy())
            elif config.model.model_type == "ensemble":
                reg_score.append(outputs.reg_score.to('cpu').detach().numpy().copy())
                logits.append(outputs.logits.to('cpu').detach().numpy().copy())

        if config.model.model_type == "hybrid":
            reg_output = np.concatenate(reg_score).tolist()
            logits = np.concatenate(logits).tolist()
            hidden_states = np.concatenate(hidden_states).tolist()
            return {"reg_output":reg_output, "logits":logits, "hidden_states":hidden_states}
        elif config.model.model_type == "classification":
            logits = np.concatenate(logits).tolist()
            hidden_states = np.concatenate(hidden_states).tolist()
            return {"logits":logits, "hidden_states":hidden_states}
        elif config.model.model_type == "gaussianregression":
            pred_score = np.concatenate(reg_score).tolist()
            pred_lnvar = np.concatenate(reg_lnvar).tolist()
            hidden_states = np.concatenate(hidden_states).tolist()
            return {"pred_score":pred_score, "pred_lnvar":pred_lnvar, "hidden_states":hidden_states}
        
    elif config.model.model_type == "ensemble":
        all_hidden_states = []
        all_reg_scores = []
        all_logits = []

        for m in model:
            m = m.cuda()
            m.eval()
            hidden_states = []
            reg_scores = []
            logits = []

            for step, inputs in enumerate(test_dataloader):
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = m(**inputs, output_hidden_states=True)
                hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
                reg_scores.append(outputs.reg_score.to('cpu').detach().numpy().copy())
                logits.append(outputs.logits.to('cpu').detach().numpy().copy())
            
            all_hidden_states.append(np.concatenate(hidden_states).tolist())
            all_reg_scores.append(np.concatenate(reg_scores).tolist())
            all_logits.append(np.concatenate(logits).tolist())
        
        return {"reg_output": all_reg_scores, "logits": all_logits, "hidden_states": all_hidden_states}

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
        int_preds = np.round(np.array(eval_results['reg_output']).reshape(-1) * (high - low))
        conf = [ps[int(i)] for ps, i in zip(probs, int_preds)]
        return int_preds, np.array(conf)
    elif config.model.model_type == 'classification':
        probs = softmax(eval_results['logits'], axis=1)
        int_preds = np.argmax(probs, axis=1)
        conf = [ps[int(i)] for ps, i in zip(probs, int_preds)]
        return int_preds, np.array(conf) 
    elif config.model.model_type == 'gaussianregression':
        int_preds = np.round(np.array(eval_results['pred_score']).reshape(-1) * (high - low))
        conf = -np.array(eval_results['pred_lnvar'])
        return int_preds, np.array(conf)
    elif config.model.model_type == 'ensemble':
        int_preds = np.round(np.mean(np.array(eval_results['reg_output']), axis=0).reshape(-1) * (high - low))
        all_probs = [softmax(l, axis=1) for l in eval_results['logits']]
        mean_probs = np.mean(all_probs, axis=0)
        conf = [ps[int(i)] for ps, i in zip(mean_probs, int_preds)]
        return int_preds, np.array(conf)


def eval_metric(config, eval_results, metric):
    if config.data.task_name == 'riken':
        high = upper_score_dic[config.data.question_id_suff][config.data.score_id]
        low = 0
    elif config.data.task_name == 'asap':
        low, high = asap_ranges[config.data.prompt_id]

    int_preds, conf = calc_predscore_conf(config, eval_results)
    if config.use_trustscore == True:
        conf = np.array(eval_results["trust_score"])

    if metric == 'qwk':
        return cohen_kappa_score(np.array(eval_results['true_labels']) + low, int_preds+low, labels = list(range(low, high + 1)), weights='quadratic')
    elif metric == 'corr':
        return np.corrcoef(np.array(eval_results['true_labels']) + low, int_preds+low)[0][1]
    elif metric == 'roc':
        errors = (np.array(eval_results['true_labels']) != int_preds).astype('int32')
        return roc_auc_score(errors, -conf)
    elif metric == 'rpp':
        squared_error = ((np.array(eval_results['true_labels']) + low) - (int_preds+low)) ** 2
        return calc_rpp(conf=conf, squared_error=squared_error)