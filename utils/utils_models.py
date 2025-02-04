from utils.score_range import upper_score_dic, asap_ranges

def create_model(num_labels, model_args, data_args, ue_args, config):

    model_base_name = model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(
        model_base_name,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
        model_type=model_args.model_type,
    )