import pandas as pd
import numpy as np
import json
from datasets import Dataset, DatasetDict

from utils.score_range import upper_score_dic, asap_ranges

def load_data(config, model_type):
    if "asap" in config.task_name:
        datasets = load_asap(config, model_type)
    elif "riken" in config.task_name:
        datasets = load_riken(config, model_type)
    return datasets

def load_asap(config, model_type):
    low, high = asap_ranges[config.prompt_id]

    train_datapath = config.data_path + f'/fold_{config.fold}' + '/train.tsv'
    train_dataf = pd.read_table(train_datapath, sep='\t')
    train_p = train_dataf[train_dataf["essay_set"] == config.prompt_id]
    train_x = train_p["essay"].tolist()
    train_y = get_model_friendly_scores(model_type, np.array(train_p["domain1_score"]), high, low).tolist()

    validation_datapath = config.data_path + f'/fold_{config.fold}' + '/dev.tsv'
    validation_dataf = pd.read_table(validation_datapath, sep='\t')
    validation_p = validation_dataf[validation_dataf["essay_set"] == config.prompt_id]
    validation_x = validation_p["essay"].tolist()
    validation_y = get_model_friendly_scores(model_type, np.array(validation_p["domain1_score"]), high, low).tolist()

    test_datapath = config.data_path + f'/fold_{config.fold}' + '/test.tsv'
    test_dataf = pd.read_table(test_datapath, sep='\t')
    test_p = test_dataf[test_dataf["essay_set"] == config.prompt_id]
    test_x = test_p["essay"].tolist()
    test_y = get_model_friendly_scores(model_type, np.array(test_p["domain1_score"]), high, low).tolist()

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_x, "label": train_y}),
            "validation": Dataset.from_dict({"text": validation_x, "label": validation_y}),
            "test": Dataset.from_dict({"text": test_x, "label": test_y}),
        }
    )
    return datasets


def load_riken(config, model_type):
    high = upper_score_dic[config.question_id_suff][config.score_id]
    low = 0
    #/${sas.prompt_id}/${sas.question_id}_data/${sas.prompt_id}_${sas.question_id}_fold${training.fold}/train_data.json
    print(config)
    train_datapath = config.data_path + '/train_data.json'
    with open(train_datapath) as f:
        train_dataf = json.load(f)
    train_x = [row['mecab'].replace(' ','') for row in train_dataf]
    score_list = [row[config.data.score_id] for row in train_dataf]
    train_y = get_model_friendly_scores(model_type, np.array(score_list), high, low).tolist()

    validation_datapath = config.data_path + '/dev_data.json'
    with open(validation_datapath) as f:
        validation_dataf = json.load(f)
    validation_x = [row['mecab'].replace(' ','') for row in validation_dataf]
    score_list = [row[config.data.score_id] for row in validation_dataf]
    validation_y = get_model_friendly_scores(model_type, np.array(score_list), high, low).tolist()
    
    test_datapath = config.data_path + '/test_data.json'
    with open(test_datapath) as f:
        test_dataf = json.load(f)
    test_x = [row['mecab'].replace(' ','') for row in test_dataf]
    score_list = [row[config.data.score_id] for row in test_dataf]
    test_y = get_model_friendly_scores(model_type, np.array(score_list), high, low).tolist()

    
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_x, "label": train_y}),
            "validation": Dataset.from_dict({"text": validation_x, "label": validation_y}),
            "test": Dataset.from_dict({"text": test_x, "label": test_y}),
        }
    )
    return datasets

def get_model_friendly_scores(model_type, score_array, high, low):
    if model_type == "regression":
        score_array = (np.array(score_array) - low) / (high - low)
    else:
        score_array = score_array - low
    return score_array        