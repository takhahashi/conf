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
from utils.score_range import upper_score_dic, asap_ranges
from utils.utils_models import create_model

log = logging.getLogger(__name__)


def compute_metrics(is_regression, metric, label_num, p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(np.round(p.predictions[1] * (label_num - 1))) if is_regression else np.argmax(preds, axis=1)
    
    result = metric.compute(predictions=preds, references=p.label_ids)

    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def train_eval_glue_model(config, training_args, data_args, work_dir=None):
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
    num_labels = high - low

    ################ Loading model #######################

    model, tokenizer = create_model(num_labels, model_args, data_args, config)

    ################ Preprocessing the dataset ###########

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)  # 512は適宜変更

    # datasets に対してトークナイザーを適用
    datasets = datasets.map(tokenize_function, batched=True)

    ################### Training ####################################


    #metric_fn = lambda p: compute_metrics(config.model.model_type, metric, num_labels, p)

    #################### Training ##########################
    trainer = get_trainer(
        model,
        training_args,
        datasets["train"],
        datasets["validation"],
        metric_fn=None,
        data_collator = data_collator,
        callbacks=None,
    )
    if model_args.model_type == 'hybrid':
        trainer.add_callback(HybridModelCallback(hb_model=model, trainer=trainer)) 
    
    trainer.train(
        model_path=model_args.model_name_or_path
        if os.path.isdir(model_args.model_name_or_path)
        else None
    )


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
    print(config)
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())
    print("==============")
    print(config.training)
    print("==============")
    """
    if config.model.model_type == 'hybrid':
        args_train = update_config(HybridTrainingArgs, config.training)
    else:
    """
    args_train = config.training
    
    args_data = config.data

    train_eval_glue_model(config, args_train, args_data, auto_generated_dir)



if __name__ == "__main__":
    main()