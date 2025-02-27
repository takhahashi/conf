import torch
import numpy as np
from tqdm import tqdm
import time
from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance,
)

import logging
import pdb
import copy

log = logging.getLogger()


class UeEstimatorMahalanobis:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        self.use_pca = "use_pca" in self.ue_args.keys() and self.ue_args.use_pca
        self.use_tanh = "use_tanh" in self.ue_args.keys() and self.ue_args.use_tanh
        self.use_encoder_feats = (
            "use_encoder_feats" in self.ue_args.keys()
            and self.ue_args.use_encoder_feats
        )
        self.return_train_preds = (
            "return_train_preds" in self.ue_args.keys()
            and self.ue_args.return_train_preds
        )

        log.info(
            "****************Start fitting covariance and centroids **************"
        )

        if y is None:
            y = self._exctract_labels(X)

        if self.return_train_preds:
            self.train_preds = self._exctract_preds(X)

        self._replace_model_head()
        X_features = self._exctract_features(X)
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covariance = self._fit_covariance(X_features, y)

        self.fit_all_md_versions = (
            "fit_all_md_versions" in self.ue_args.keys()
            and self.ue_args.fit_all_md_versions
        )
        self.return_features = (
            "return_features" in self.ue_args.keys() and self.ue_args.return_features
        )
        if self.fit_all_md_versions:
            self.train_centroid = self._fit_centroids(X_features, y, class_cond=False)
            self.train_covariance = self._fit_covariance(
                X_features, y, class_cond=False
            )

        if self.return_features or self.return_train_preds:
            self.train_features = X_features
            self.train_labels = y

        log.info("**************Done.**********************")

    def _fit_covariance(self, X, y, class_cond=True):
        pdb.set_trace()
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)

    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model


        if "distilbert" in self.config.model.model_name_or_path:
            if self.use_encoder_feats:
                model.pre_classifier = torch.nn.Identity()
                model.pre_classifier_activation = torch.nn.Identity()
                model.dropout = torch.nn.Identity()
            if self.use_tanh:
                model.pre_classifier_activation = torch.nn.Tanh()
        elif "deberta" in self.config.model.model_name_or_path:
            if self.use_tanh:
                model.pooler.activation = torch.nn.Tanh()

    def _exctract_labels(self, X):
        return np.asarray([example["label"] for example in X])

    def _exctract_preds(self, X):
        cls = self.cls
        model = self.cls._auto_model
        X_cp = copy.deepcopy(X)

        try:
            X_cp = X_cp.remove_columns("label")
        except:
            X_cp.dataset = X_cp.dataset.remove_columns("label")

        X_preds = cls.predict(X_cp, apply_softmax=True, return_preds=False)[0]
        return X_preds

    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model

        try:
            X = X.remove_columns("label")
        except:
            X.dataset = X.dataset.remove_columns("label")

        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        preds, probs = self.cls.predict(X)[:2]
        X_features = np.array(X_features, np.float64)

        return X_features

    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Compute MD with fitted covariance and centroids **************"
        )

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        if self.use_pca:
            X_features = self.pca.transform(X_features)
        end = time.time()

        eval_results = {}

        md, inf_time = mahalanobis_distance(
            None,
            None,
            X_features,
            self.class_cond_centroids,
            self.class_cond_covariance,
        )

        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()

        if self.return_train_preds:
            train_md, _ = mahalanobis_distance(
                None,
                None,
                self.train_features,
                self.class_cond_centroids,
                self.class_cond_covariance,
            )

            eval_results["train_mahalanobis_distance"] = train_md.tolist()
            eval_results["train_preds"] = self.train_preds.tolist()
            eval_results["train_labels"] = self.train_labels.flatten().tolist()

        eval_results["ue_time"] = sum_inf_time
        eval_results["inf_time_only"] = inf_time
        log.info(f"UE time: {sum_inf_time}")

        if self.fit_all_md_versions:
            md_relative = mahalanobis_distance_relative(
                None,
                None,
                X_features,
                self.class_cond_centroids,
                self.class_cond_covariance,
                self.train_centroid,
                self.train_covariance,
            )

            md_marginal = mahalanobis_distance_marginal(
                None, None, X_features, self.train_centroid, self.train_covariance
            )

            eval_results["mahalanobis_distance_relative"] = md_relative.tolist()
            eval_results["mahalanobis_distance_marginal"] = md_marginal.tolist()

        if self.return_features:
            eval_results["train_features"] = self.train_features.flatten().tolist()
            eval_results["train_labels"] = self.train_labels.flatten().tolist()
            eval_results["eval_features"] = X_features.flatten().tolist()

        log.info("**************Done.**********************")
        return eval_results