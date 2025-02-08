
import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from typing import Optional

from utils.model import HybridOutput

@dataclass
class HybridTrainingArgs(TrainingArguments):
    lamb: Optional[float] = None
    margin: Optional[float] = None
    lamb_intra: Optional[float] = None


def get_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    metric_fn,
    data_collator=None,
    callbacks=None,
) -> "Trainer":

    if model.__class__.__name__ == "HybridBert":
        trainer = HybridTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    return trainer


def compute_loss_metric(
    hiddens, labels, loss, num_labels, margin, lamb_intra, lamb, unpad=False, probabilities=None,
):
    """Computes regularization term for loss with Metric loss"""
    class_num = num_labels
    start_idx = 0 if class_num == 2 else 1
    # TODO: define represent, target and margin
    # Get only sentence representaions
    (
        loss_intra,
        loss_inter,
    ) = multiclass_metric_loss_fast_optimized(  # multiclass_metric_loss_fast(
        hiddens,
        labels,
        margin=margin,
        class_num=class_num,
        probabilities=probabilities,
    )
    loss_metric = lamb_intra * loss_intra[0] + lamb * loss_inter[0]
    loss += loss_metric
    return loss

def multiclass_metric_loss_fast_optimized(represent, target, probabilities, class_num, margin):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]

    indices = []

    for class_idx in range(0, class_num):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0

    cls_repr = {}
    cls_p = {}
    for i in range(class_num):
        indices_i = indices[i]
        curr_repr = represent[indices_i]
        curr_p = probabilities[indices_i]

        if len(curr_repr) > 0:
            curr_p = torch.tensor([1., 1.])
            curr_repr = torch.tensor([[1.,1.],[2.,2.]])
            cls_repr[i] = curr_repr
            cls_p[i] = curr_p
            p_matrix = curr_p.unsqueeze(1) * curr_p
            triangle_matrix = torch.triu(
                p_matrix * (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
            )

            loss_intra += torch.sum(1 / dim * (triangle_matrix**2))
            num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2
            print("==========curr_p==========")
            print(curr_p)
            print("==========p_matrix==========")
            print(p_matrix)
            print("==========euclid==========")
            print((curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1))
            print("==========triangle_matrix==========")
            print(triangle_matrix)
            print("==========before_sum==========")
            dim=2
            
            print(1 / dim * (triangle_matrix**2))
            exit()

    batch_labels = list(cls_repr.keys())
    bs = represent.shape[0]
    for n, j in enumerate(batch_labels):
        curr_repr = cls_repr[j]
        curr_p = cls_p[j]
        for l, k in enumerate(batch_labels[n + 1 :]):
            
            p_matrix = (curr_p.unsqueeze(1) * cls_p[k]).flatten()
            matrix = ((curr_repr.unsqueeze(1) - cls_repr[k]).norm(2, dim=-1)).flatten()

            loss_inter += torch.sum(
                torch.clamp(margin * p_matrix - 1 / dim * (matrix**2), min=0)
            )
            num_inter += cls_repr[k].shape[0] * curr_repr.shape[0]

    if num_intra > 0:
        loss_intra = loss_intra / num_intra
    if num_inter > 0:
        loss_inter = loss_inter / num_inter


    return loss_intra, loss_inter

class HybridTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lamb = getattr(kwargs["args"], "lamb")
        self.margin = getattr(kwargs["args"], "margin")
        self.lamb_intra = getattr(kwargs["args"], "lamb_intra")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs, output_hidden_states=True)

        probabilities = None
        hiddens = outputs.hidden_states[-1][:, 0, :]
        logits = outputs.logits

        softmax_probabilities = F.softmax(logits, dim=-1)
        reg_output = outputs.reg_output
        reg_pred_int = np.round((model.config.num_labels - 1) * reg_output.to('cpu').detach().view(-1).numpy().copy())
        probabilities = softmax_probabilities[list(range(len(softmax_probabilities))), reg_pred_int]

        loss = outputs.loss
        
        del outputs
        torch.cuda.empty_cache()
        
        loss = compute_loss_metric(
            hiddens,
            labels,
            loss,
            model.config.num_labels,
            self.margin,
            self.lamb_intra,
            self.lamb,
            probabilities=probabilities,
        )


        if isinstance(logits, tuple):
            return (loss,) + logits if return_outputs else loss
        else:
            return (loss, logits) if return_outputs else loss

class HybridModelCallback(TrainerCallback):
    def __init__(self, hb_model, trainer):
        super().__init__()
        self.hb_model = hb_model
        self.trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.hb_model.lsb.update()
        for k, v in self.hb_model.lsb.loss_log.items():
            scaled_loss = self.hb_model.diff_weights[k].to('cpu').detach().numpy().copy() * self.hb_model.scale_weights[k].to('cpu').detach().numpy().copy() * v[-1]
            each_task_loss = v[-1]
            self.trainer.log({f"{k}_scaled_loss": scaled_loss, f"{k}_loss":each_task_loss})