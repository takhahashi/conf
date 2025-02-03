import os
import hydra
import logging
import pdb
from train_config import TrainingArgs

log = logging.getLogger(__name__)

if __name__ == "__main__":

    @hydra.main(
        config_name='configs/test',
    )
    def main(config):
        print(config)
        print(os.getcwd())

    main()