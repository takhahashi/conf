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
from transformers import (
    EvalPrediction,
)
from utils.utils_train import (
    HybridTrainingArgs,
    get_trainer,
    HybridModelCallback,
)
from utils.utils_eval import (
    evaluate_model,
)
from utils.score_range import upper_score_dic, asap_ranges
from utils.utils_models import create_model

log = logging.getLogger(__name__)

def eval_model(config, data_args, work_dir=None):
    torch.manual_seed(config.model.id)
    log.info(f"config:{config}")

    model_args = config.model

    ############### Loading dataset ######################

    log.info("Load dataset.")
    datasets = load_data(data_args, config.model.model_type)
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

    eval_resutls = evaluate_model(config, model, datasets)


def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old

@hydra.main(
    config_path='configs',
    config_name='training',
)
def main(config):
    #os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    eval_model(config, config.data, auto_generated_dir)



if __name__ == "__main__":
    main()