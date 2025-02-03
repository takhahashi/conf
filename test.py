import os
import hydra
import logging
import pdb
from train_config import TrainingArgs

log = logging.getLogger(__name__)

if __name__ == "__main__":

    @hydra.main(
        config_path='configs',
        config_name='test',
    )
    def main(config):
        print(config)
        print(os.getcwd())
        print

    main()