import os
from pathlib import Path
import hydra
import logging
from transformers import TrainingArguments
import torch
import gpytorch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.utils_data import (
    load_data,
    data_collator,
)
from utils.dataset import CustomDataset
from transformers import (
    EvalPrediction,
)
from utils.utils_train import (
    HybridTrainingArgs,
    get_trainer,
    HybridModelCallback,
)
from utils.model import GPModel
from utils.score_range import upper_score_dic, asap_ranges
from utils.utils_models import create_model
import wandb

log = logging.getLogger(__name__)


def compute_metrics(is_regression, metric, label_num, p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(np.round(p.predictions[1] * (label_num - 1))) if is_regression else np.argmax(preds, axis=1)
    
    result = metric.compute(predictions=preds, references=p.label_ids)

    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def train_model(config, training_args, data_args, work_dir=None):
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
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512) 

    # datasets に対してトークナイザーを適用
    datasets = datasets.map(tokenize_function, batched=True)

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
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

def train_gp(config, training_args, data_args, work_dir=None):
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

    encoder_model, tokenizer = create_model(num_labels, model_args, data_args, config)

    ################ Preprocessing the dataset ###########

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512) 

    # datasets に対してトークナイザーを適用
    datasets = datasets.map(tokenize_function, batched=True)
    
    train_dataset = CustomDataset(datasets['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    if config.model.model_type != "ensemble":
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
    GPmodel = GPModel(train_x, train_y, likelihood)

    epoch = training_args.num_train_epochs
    GPmodel.train()
    likelihood.train()
    GPmodel.covar_module.base_kernel.lengthscale = np.linalg.norm(train_x[0].numpy() - train_x[1].numpy().T) ** 2 / 2

    optimizer = torch.optim.Adam([
        {'params': GPmodel.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=training_args.learning_rate)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, GPmodel)

    for i in range(epoch):
        optimizer.zero_grad()
        output = GPmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        print('Iter %d/%d - Loss: %.3f lengthscale: %.3f noise: %.3f' % (
            i+1, epoch, loss.item(),
            GPmodel.covar_module.base_kernel.lengthscale.item(),
            GPmodel.likelihood.noise.item()
        ))
    model_savepath = Path(config.training.output_dir) / "gp_model"
    model_savepath.parent.mkdir(parents=True)
    torch.save(GPmodel.state_dict(), model_savepath)

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

    wandb.init(
        project=config.training.wandb_project,
        name=config.training.wandb_runname,
    )
    

    if config.model.model_type == 'hybrid':
        args_train = update_config(HybridTrainingArgs(output_dir=config.training.output_dir, report_to='wandb'), config.training)
    else:
        args_train = update_config(TrainingArguments(output_dir=config.training.output_dir, report_to='wandb'), config.training)

    if config.model.model_type == 'gp':
        train_gp(config, args_train, config.data, auto_generated_dir)
    else:
        train_model(config, args_train, config.data, auto_generated_dir)



if __name__ == "__main__":
    main()