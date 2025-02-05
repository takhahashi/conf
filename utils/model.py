import numpy as np
from transformers.modeling_outputs import (
    ModelOutput,
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    )
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
)
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Tuple
import torch

@dataclass
class HybridOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    reg_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class ScaleDiffBalance:
  def __init__(self, task_names, priority=None, beta=1.):
    self.task_names = task_names
    self.num_tasks = len(self.task_names)
    self.task_priority = {}
    if priority is not None:
        for k, v in priority.items():
           self.task_priority[k] = v
    else:
        for k in self.task_names:
          self.task_priority[k] =  1/self.num_tasks
    self.all_loss_log = []
    self.loss_log = defaultdict(list)
    self.beta = beta
    self.all_batch_loss = 0
    self.each_task_batch_loss = {}
    for k in self.task_names:
        self.each_task_batch_loss[k] = 0
    self.batch_count = 0
  
  def update(self, *args, **kwargs):
    self.all_loss_log = np.append(self.all_loss_log, self.all_batch_loss/self.batch_count)
    self.all_batch_loss = 0
    for k, v in self.each_task_batch_loss.items():
       self.loss_log[k] = np.append(self.loss_log[k], v/self.batch_count)
       self.each_task_batch_loss[k] = 0
    self.batch_count = 0
  
  def __call__(self, *args, **kwargs):
    self.batch_count += 1
    scale_weights = self._calc_scale_weights()
    diff_weights = self._calc_diff_weights()
    alpha = self._calc_alpha(diff_weights)
    all_loss = 0
    for k, each_loss in kwargs.items():
       all_loss += scale_weights[k] * diff_weights[k] * each_loss
       self.each_task_batch_loss[k] += each_loss.to('cpu').detach().numpy().copy()
    if len(self.all_loss_log) < 1:
      pre_loss = 0
    else:
      pre_loss = self.all_loss_log[-1]
    self.all_batch_loss += (alpha * all_loss).to('cpu').detach().numpy().copy()
    return alpha * all_loss, scale_weights, diff_weights, alpha, pre_loss
  
  def _calc_scale_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 1:
      for k, v in self.task_priority.items():
         w_dic[k] = torch.tensor(v).cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         task_priority = self.task_priority[k]
         w_dic[k] = torch.tensor(self.all_loss_log[-1]*task_priority/each_task_loss_arr[-1]).cuda()
    return w_dic
  
  def _calc_diff_weights(self):
    w_dic = {}
    if len(self.all_loss_log) < 2:
      for k, _ in self.task_priority.items():
         w_dic[k] = torch.tensor(1.).cuda()
    else:
      for k, each_task_loss_arr in self.loss_log.items():
         w_dic[k] = torch.tensor(((each_task_loss_arr[-1]/each_task_loss_arr[-2])/(self.all_loss_log[-1]/self.all_loss_log[-2]))**self.beta).cuda()
    return w_dic
  
  def _calc_alpha(self, diff_weights):
    if len(self.all_loss_log) < 2:
      return torch.tensor(1.).cuda()
    else:
      tmp = 0
      for k, v in self.task_priority.items():
         tmp += torch.tensor(v).cuda() * diff_weights[k]
      return (1/tmp).cuda()
    

class HybridBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.lsb = ScaleDiffBalance(task_names=['regression', 'classification'])
        self.scale_weights = {}
        self.diff_weights = {}
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        regressor_output = self.sigmoid(self.regressor(pooled_output))

        loss = None
        if labels is not None:
            ########regression loss########
            reg_labels = labels.view(-1) / (self.num_labels - 1)
            loss_fct = MSELoss()
            r_loss = loss_fct(regressor_output.view(-1), reg_labels)           
            
            ########classification loss#########
            loss_fct = CrossEntropyLoss()
            c_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss, s_wei, diff_wei, alpha, pre_loss = self.lsb(regression=r_loss, classification=c_loss)
            self.scale_weights = s_wei
            self.diff_weights = diff_wei

            

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        

        return HybridOutput(
            loss=loss,
            logits=logits,
            reg_output=regressor_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )