import os
import hydra
import logging
from transformers import TrainingArguments
import torch
import numpy as np
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
from utils.ue_estimater import create_ue_estimator
from pathlib import Path
import json

log = logging.getLogger(__name__)

def eval_model(config, data_args):
    torch.manual_seed(config.model.id)
    log.info(f"config:{config}")

    model_args = config.model

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

    model, tokenizer = create_model(num_labels, model_args, data_args, config)

    ################ Preprocessing the dataset ###########

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)  # 512は適宜変更

    # datasets に対してトークナイザーを適用
    datasets = datasets.map(tokenize_function, batched=True)

    ################### Evaluate ####################################

    eval_results = evaluate_model(config, model, datasets)
    eval_results["true_labels"] = [example["label"] for example in datasets['test']]

    if config.use_trustscore:
        train_dataset = datasets["train"]
        eval_dataset = datasets["test"]
        true_labels = [example["label"] for example in eval_dataset]
        
        ue_estimator = create_ue_estimator(
            model,
            config.ue,
            train_dataset=datasets["train"],
            config=config,
        )

        ue_estimator.fit_ue(X=train_dataset, X_test=eval_dataset)

        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    eval_results["qwk"] = eval_metric(config, eval_results, 'qwk')
    eval_results["corr"] = eval_metric(config, eval_results, 'corr')
    eval_results["roc"] = eval_metric(config, eval_results, 'roc')
    eval_results["rpp"] = eval_metric(config, eval_results, 'rpp')

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
    config_name='eval',
)
def main(config):
    #os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    eval_model(config, config.data)



if __name__ == "__main__":
    main()