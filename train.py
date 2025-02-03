@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    args_train = TrainingArgsWithLossCoefs(
        output_dir=auto_generated_dir,
        reg_type=config.ue.get("reg_type", "reg-curr"),
        lamb=config.ue.get("lamb", 0.01),
        margin=config.ue.get("margin", 0.05),
        lamb_intra=config.ue.get("lamb_intra", 0.01),
    )
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(task_name=config.data.task_name)
    args_data = update_config(args_data, config.data)

    if config.do_train and not config.do_eval:
        filename = "pytorch_model.bin"
    else:
        filename = "dev_inference.json"

    config.ue.use_cache=False

    if not os.path.exists(Path(auto_generated_dir) / filename):
        if config.model.model_type == 'gp':
            train_eval_gp_model(config, args_train, args_data, auto_generated_dir)
        else:
            train_eval_glue_model(config, args_train, args_data, auto_generated_dir)
    else:
        log.info(f"Result file: {auto_generated_dir}/{filename} already exists \n")
    wandb.finish()


if __name__ == "__main__":
    main()