import os
import hydra
import logging
from transformers import TrainingArguments
import gpytorch
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader
from utils.utils_data import (
    load_data,
    data_collator,
)
from utils.utils_eval import (
    evaluate_model,
    eval_metric
)
from utils.score_range import upper_score_dic, asap_ranges
from utils.utils_models import create_model
from utils.model import GPModel
from utils.ue_estimater import create_ue_estimator
from utils.dataset import CustomDataset
from ue4nlp.ue_estimator_trustscore import UeEstimatorTrustscore
from pathlib import Path
import json

log = logging.getLogger(__name__)

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

def eval_model(config, data_args):
    log.info(f"config:{config}")
    ############### Loading dataset ######################

    log.info("Load dataset.")
    datasets = load_data(data_args)
    log.info("Done with loading the dataset.")

    if data_args.task_name == 'riken':
        high = upper_score_dic[data_args.question_id_suff][data_args.score_id]
        low = 0
    elif data_args.task_name == 'asap':
        low, high = asap_ranges[data_args.prompt_id]
    num_labels = high - low + 1

    ################ Loading model #######################

    encoder_model, tokenizer = create_model(num_labels, config.encoder_model, data_args, config)

    ################ Preprocessing the dataset ###########

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)  # 512は適宜変更

    # datasets に対してトークナイザーを適用
    datasets = datasets.map(tokenize_function, batched=True)

    ################### Evaluate ####################################
    train_dataset = CustomDataset(datasets['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    encoder_model = encoder_model.cuda()
    encoder_model.eval()
    hidden_states = []
    labels = []
    for step, inputs in enumerate(train_dataloader):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = encoder_model(**inputs, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        labels.append(inputs["labels"].to('cpu').detach().numpy().copy())
    hidden_states = np.concatenate(hidden_states).tolist()
    labels = np.concatenate(labels)
    train_x = torch.FloatTensor(hidden_states)
    train_y = torch.FloatTensor(labels)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GPModel(train_x, train_y, likelihood)

    test_dataset = CustomDataset(datasets['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    
    hidden_states = []
    labels = []
    for step, inputs in enumerate(test_dataloader):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = encoder_model(**inputs, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        labels.append(inputs["labels"].to('cpu').detach().numpy().copy())
    hidden_states = np.concatenate(hidden_states).tolist()
    labels = np.concatenate(labels)
    test_x = torch.FloatTensor(hidden_states)
    test_y = torch.FloatTensor(labels)

    gp.load_state_dict(torch.load(config.gp_model.model_name_or_path))
    likelihood.eval()
    gp.eval()    
    predictions = gp(test_x)
    mean = predictions.mean.cpu().detach().numpy()
    std = predictions.stddev.cpu().detach().numpy()
    eval_results = {'true_labels':list(labels.astype('float64')), 'score':list(mean.astype('float64')), 'std':list(std.astype('float64'))}

    int_preds = np.round(np.array(eval_results['score']))
    conf = -np.array(eval_results['std'])
    eval_results['qwk'] = cohen_kappa_score(np.array(eval_results['true_labels']) + low, int_preds+low, labels = list(range(low, high + 1)), weights='quadratic')
    eval_results['corr'] = np.corrcoef(np.array(eval_results['true_labels']) + low, int_preds+low)[0][1]

    errors = (np.array(eval_results['true_labels']) != int_preds).astype('int32')
    eval_results['roc'] = roc_auc_score(errors, -conf)

    squared_error = ((np.array(eval_results['true_labels']) + low) - (int_preds+low)) ** 2
    eval_results['rpp'] = calc_rpp(conf=conf, squared_error=squared_error)


    resut_savepath = Path(config.result_savepath) / "test_inference.json"
    resut_savepath.parent.mkdir(parents=True, exist_ok=True)
    with open(resut_savepath, "w") as res:
        json.dump(eval_results, res)

def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old

@hydra.main(
    config_path='configs',
    config_name='gp_eval',
)
def main(config):
    #os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    eval_model(config, config.data)



if __name__ == "__main__":
    main()