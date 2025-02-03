from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class CustomTrainingArgs(TrainingArguments):
    lamb: float
    margin: float
    lamb_intra: float