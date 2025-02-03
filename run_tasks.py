import os
import hydra
import logging
import pdb

log = logging.getLogger(__name__)

if __name__ == "__main__":

    @hydra.main(
        config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
        config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
    )
    def main(config):
        auto_generated_dir = os.getcwd()
        log.info(f"Work dir: {auto_generated_dir}")
        os.chdir(hydra.utils.get_original_cwd())
        print(os.getcwd())


        #run_glue_for_model_series_fast(config, auto_generated_dir)

    main()