import os
import hydra
import logging
from transformers import TrainingArguments
import torch
from utils.utils_data import (
    load_data,
)
from utils.utils_train import CustomTrainingArgs

log = logging.getLogger(__name__)

def train_eval_glue_model(config, training_args, data_args, work_dir=None):
    torch.manual_seed(config.model.id)
    log.info(f"config:{config}")

    model_args = config.model

    ############### Loading dataset ######################

    log.info("Load dataset.")
    datasets = load_data(data_args)
    log.info("Done with loading the dataset.")
    print(datasets["train"]["label"][:10])
    exit()

    # Labels
    if data_args.task_name in glue_datasets:
        label_list = datasets["train"].features["label"].names
    else:
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism

    if config.task_name == 'asap':
        low, high = get_score_range(config.task_name, config.prompt_id)
        num_labels = high - low + 1
    elif config.task_name == 'riken':
        high = upper_score_dic[config.prompt_id][config.score_id]
        low = 0
        num_labels = high - low + 1
    else:
        num_labels = len(label_list)
    log.info(f"Number of labels: {num_labels}")

    ################ Loading model #######################

    model, tokenizer = create_model(num_labels, model_args, data_args, ue_args, config)

    ################ Preprocessing the dataset ###########

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    sentence2_key = (
        None
        if (config.task_name in ["bios", "trustpilot", "jigsaw_race", "sepsis_ethnicity", "asap", "riken"])
        else sentence2_key
    )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            log.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    f_preprocess = lambda examples: preprocess_function(
        label_to_id, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    datasets = datasets.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if "idx" in datasets.column_names["train"]:
        datasets = datasets.remove_columns("idx")

    ################### Training ####################################
    if config.reset_params:
        reset_params(model)

    if ue_args.dropout_type == "DC_MC":
        convert_dropouts(model, ue_args)

    train_dataset = datasets["train"]
    train_indexes = list(range(len(train_dataset)))
    calibration_dataset = None
    eval_dataset = datasets["validation"]

    log.info(f"Training dataset size: {len(train_dataset)}")
    log.info(f"Eval dataset size: {len(eval_dataset)}")
    
    test_dataset = datasets["test"]

    metric = load_metric(
        "accuracy", keep_in_memory=True, cache_dir=config.cache_dir
    )

    is_regression = False
    metric_fn = lambda p: compute_metrics(is_regression, metric, num_labels, p)

    if config.do_train:


        #training_args.warmup_steps = int(
        #    training_args.warmup_ratio  # TODO:
        #    * len(train_dataset)
        #    * training_args.num_train_epochs
        #    / training_args.train_batch_size
        #)
        #log.info(f"Warmup steps: {training_args.warmup_steps}")
        #training_args.logging_steps = training_args.warmup_steps


        training_args.weight_decay_rate = training_args.weight_decay

    data_collator = simple_collate_fn
    training_args = update_config(training_args, {'fp16':True})
    
    use_sngp = ue_args.ue_type == "sngp"
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective
    
    training_args = update_config(training_args, {'load_best_model_at_end':True})
    training_args = update_config(training_args, {'eval_strategy':'epoch'})
    training_args = update_config(training_args, {'metric_for_best_model':'eval_loss'})
    training_args = update_config(training_args, {'save_strategy':'epoch'})
    if "patience" in config.training.keys():
        earlystopping = EarlyStoppingCallback(early_stopping_patience=int(config.training.patience))
        callbacks = [earlystopping]
    else:
        callbacks = None
    #################### Training ##########################
    trainer = get_trainer(
        model_args.model_type,
        use_selective,
        use_sngp,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        metric_fn,
        data_collator = data_collator,
        callbacks=callbacks,
    )
    if model_args.model_type == 'hybrid':
        trainer.add_callback(HybridModelCallback(hb_model=model, trainer=trainer)) 
    if ue_args.reg_type == 'ExpEntropyLearning':
        trainer.add_callback(ExpEntCallback(trainer=trainer)) 

    
    if config.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # Rewrite the optimal hyperparam data if we want the evaluation metrics of the final trainer
        if config.do_eval:
            evaluation_metrics = trainer.evaluate()
        if work_dir != None:
            trainer.save_model(work_dir)
            tokenizer.save_pretrained(work_dir)



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

    if config.model.model_type == 'hybrid':
        args_train = update_config(CustomTrainingArgs, config.training)
    else:
        args_train = config.training
    
    args_data = config.data

    train_eval_glue_model(config, args_train, args_data, auto_generated_dir)



if __name__ == "__main__":
    main()