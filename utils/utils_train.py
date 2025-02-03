from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional

@dataclass
class CustomTrainingArgs(TrainingArguments):
    lamb: Optional[float] = None
    margin: Optional[float] = None
    lamb_intra: Optional[float] = None