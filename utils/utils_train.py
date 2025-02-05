from dataclasses import dataclass, field
from transformers import (
    TrainingArguments,
    Trainer,
)
from typing import Optional


@dataclass
class CustomTrainingArgs(TrainingArguments):
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