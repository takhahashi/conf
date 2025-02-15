from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis
from ue4nlp.ue_estimator_trustscore import UeEstimatorTrustscore
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


def create_ue_estimator(
    model,
    ue_args,
    train_dataset,
    config=None,
):
    if ue_args.ue_type == "maha":
        return UeEstimatorMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "trust":
        return UeEstimatorTrustscore(model, ue_args, config, train_dataset)
    else:
        raise ValueError()