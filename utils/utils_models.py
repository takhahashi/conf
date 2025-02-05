from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from utils.score_range import upper_score_dic, asap_ranges
from utils.model import HybridBert
import logging

log = logging.getLogger(__name__)

def create_model(num_labels, model_args, data_args, ue_args, config):

    model_base_name = model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(
        model_base_name,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        model_type=model_args.model_type,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_base_name,
        cache_dir=config.cache_dir,
    )
    models_constructors = {
        "roberta": create_roberta,
        "deberta": create_deberta,
        "distilbert": create_distilbert,
        "bert": create_bert,
    }
    for key, value in models_constructors.items():
        if key in model_base_name:
            return (
                models_constructors[key](
                    model_config,
                    tokenizer,
                    ue_args,
                    model_base_name,
                    config,
                ),
                tokenizer,
            )
    raise ValueError(f"Cannot find model with this name or path: {base_model_name}")

def build_model(model_class, model_path_or_name, reg_type=None, **kwargs):
    if reg_type == 'label_distribution':
        return model_class.from_pretrained(model_path_or_name, reg_type, **kwargs)
    else:
        return model_class.from_pretrained(model_path_or_name, **kwargs)
    
def create_bert(
    model_config,
    model_path_or_name,
    config,
):
    model_kwargs = dict(
        from_tf=False,
        config=model_config,
        cache_dir=config.cache_dir,
    )
    if model_config.model_type == 'hybrid':
            model = build_model(
                HybridBert, model_config._name_or_path, **model_kwargs
            )
            log.info("loaded HybridBERT constraction")
    elif model_config.model_type == 'regression':
        model = build_model(
            BertForSequenceRegression, model_path_or_name, **model_kwargs
        )
        log.info("loaded RegressionBERT constraction")
    elif model_config.model_type == 'classification':
        model = build_model(
            AutoModelForSequenceClassification, model_path_or_name, **model_kwargs
        )
        log.info("loaded ClassificationBERT constraction")
    elif model_config.model_type == 'normalregression':
        model = build_model(
            BertForSequenceNormalRegression, model_path_or_name, **model_kwargs
        )
        log.info("loaded NormalRegression constraction")
    else:
        raise ValueError(f"{model_config.model_type} IS INVALID MODEL_TYPE")
    return model