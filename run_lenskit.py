import json
import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd
from lenskit import topn_pipeline, recommend, score, RecPipelineBuilder
from lenskit.basic import FallbackScorer
from lenskit.data import from_interactions_df, ItemList, ItemListCollection
from lenskit.basic.popularity import PopScorer as PopScore
from lenskit.knn.item import ItemKNNScorer as ItemItem
from lenskit.knn.user import UserKNNScorer as UserUser
from lenskit.als import ImplicitMFScorer as ImplicitMF, BiasedMFScorer as BiasedMF
from lenskit.funksvd import FunkSVDScorer as FunkSVD
from lenskit.metrics import RunAnalysis, RunAnalysisResult
from lenskit.metrics.ranking import NDCG

from algorithm_config import retrieve_configurations
import binpickle
import time
from run_utils import ndcg, hr, recall


def lenskit_load_transform(data_set_name, fold, partition):
    if partition == "train":
        data = pd.read_csv(f"./data_sets/{data_set_name}/atomic/{data_set_name}.train_split_fold_{fold}.inter",
                           header=0, sep=",")
    elif partition == "valid":
        data = pd.read_csv(f"./data_sets/{data_set_name}/atomic/{data_set_name}.valid_split_fold_{fold}.inter",
                           header=0, sep=",")
    elif partition == "test":
        data = pd.read_csv(f"./data_sets/{data_set_name}/atomic/{data_set_name}.test_split_fold_{fold}.inter",
                           header=0, sep=",")
    else:
        print(f"Partition {partition} not found.")
        return

    data.rename(columns={'user_id:token': 'user', 'item_id:token': 'item', 'rating:float': 'rating'}, inplace=True)

    return data

def lenskit_fit(mode, data_set_name, algorithm_name, algorithm_config, fold, num_samples=None, seed=None):
    setup_start_time = time.time()

    data = lenskit_load_transform(data_set_name, fold, "train")
    train = from_interactions_df(data, user_col='user', item_col='item', rating_col='rating')

    configurations = retrieve_configurations(algorithm_name=algorithm_name, num_samples=num_samples, seed=seed)
    current_configuration = configurations[algorithm_config]

    predict_rating = False

    if algorithm_name == "PopScore":
        scorer = PopScore(**current_configuration)
    elif algorithm_name == "ItemItem":
        if "rating" in data.columns:
            scorer = ItemItem(**current_configuration, feedback="explicit")
            predict_rating = True
        else:
            scorer = ItemItem(**current_configuration, feedback="implicit")
    elif algorithm_name == "UserUser":
        if "rating" in data.columns:
            scorer = UserUser(**current_configuration, feedback="explicit")
            predict_rating = True
        else:
            scorer = UserUser(**current_configuration, feedback="implicit")
    elif algorithm_name == "ImplicitMF":
        scorer = ImplicitMF(**current_configuration, rng_spec=42)
    elif algorithm_name == "BiasedMF":
        scorer = BiasedMF(**current_configuration, rng_spec=42)
        predict_rating = True
    elif algorithm_name == "FunkSVD":
        scorer = FunkSVD(**current_configuration, random_state=42)
        predict_rating = True
    else:
        print(f"Algorithm {algorithm_name} not found.")
        return

    pipeline = topn_pipeline(scorer=scorer, predicts_ratings=predict_rating)

    setup_end_time = time.time()

    limit_time = True
    fit_start_time = time.time()
    if limit_time:
        remaining_time = 1800 - int((setup_end_time - setup_start_time))
        print(f"Remaining time: {remaining_time} seconds.")
        if remaining_time < 0:
            print(f"Setup exceeded time limit.")

        if os.name == 'nt':
            pipeline.train(train)
        elif os.name == 'posix':
            def timeout_fit(signum, frame):
                raise TimeoutError("Training exceeded time limit.")

            signal.signal(signal.SIGALRM, timeout_fit)
            signal.alarm(remaining_time)
            try:
                pipeline.train(train)
            except TimeoutError:
                print("Training exceeded time limit.")
                pass
    else:
        pipeline.train(train)
    fit_end_time = time.time()

    target_path = f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/config_{algorithm_config}/fold_{fold}/"
    Path(target_path).mkdir(parents=True, exist_ok=True)
    current_time = time.time()
    pipeline_file = f"{target_path}{algorithm_name}-{current_time}.bpk"
    binpickle.dump(pipeline, pipeline_file)

    fit_log_dict = {
        "pipeline_file": pipeline_file,
        "data_set_name": data_set_name,
        "algorithm_name": algorithm_name,
        "algorithm_config_index": algorithm_config,
        "algorithm_configuration": configurations[algorithm_config],
        "fold": fold,
        # "best_validation_score": best_valid_score,
        "setup_time": setup_end_time - setup_start_time,
        "training_time": fit_end_time - fit_start_time
    }

    with open(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
              f"config_{algorithm_config}/fold_{fold}/fit_log.json", mode="w") as file:
        json.dump(fit_log_dict, file, indent=4)


def lenskit_predict(mode, data_set_name, algorithm_name, algorithm_config, fold):
    configurations = retrieve_configurations(algorithm_name=algorithm_name, num_samples=num_samples, seed=seed)

    fit_log_file = (f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
                    f"config_{algorithm_config}/fold_{fold}/fit_log.json")
    with open(fit_log_file, "r") as file:
        fit_log = json.load(file)
    pipeline_file = fit_log["pipeline_file"]

    pipeline = binpickle.load(pipeline_file)

    train_data = lenskit_load_transform(data_set_name, fold, "train")
    test_data = lenskit_load_transform(data_set_name, fold, "test")

    predict_log_dict = {
        "pipeline_file": pipeline_file,
        "data_set_name": data_set_name,
        "algorithm_name": algorithm_name,
        "algorithm_config_index": algorithm_config,
        "algorithm_configuration": configurations[algorithm_config],
        "fold": fold
    }

    if "rating" not in train_data.columns:
        unique_train_users = train_data["user"].unique()
        unique_test_users = test_data["user"].unique()
        users_to_predict = np.intersect1d(unique_test_users, unique_train_users)

        top_k_dict = {}

        start_prediction = time.time()

        for user in users_to_predict:
            predictions = recommend(pipeline=pipeline, query=user, n=20).to_df()
            top_k_dict[int(user)] = [predictions["item_id"].tolist(), predictions["score"].tolist()]

        end_prediction = time.time()

        with open(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
                  f"config_{algorithm_config}/fold_{fold}/predictions.json", "w") as file:
            json.dump(top_k_dict, file, indent=4)

        predict_log_dict.update({
            "train_users": len(unique_train_users),
            "test_users": len(unique_test_users),
            "users_to_predict": len(users_to_predict),
            "prediction_time": end_prediction - start_prediction
        })
    else:
        test_interactions = len(test_data)
        start_prediction = time.time()
        unique_items = test_data["item"].unique()
        predictions = score(pipeline=pipeline, query=test_data.drop(columns="rating"), items=unique_items)
        end_prediction = time.time()

        predictions.to_df().to_csv(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
                           f"config_{algorithm_config}/fold_{fold}/predictions.csv", index=False)

        predict_log_dict.update({
            "test_interactions": test_interactions,
            "prediction_time": end_prediction - start_prediction
        })

    with open(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
              f"config_{algorithm_config}/fold_{fold}/predict_log.json", "w") as file:
        json.dump(predict_log_dict, file, indent=4)


def lenskit_evaluate(mode, data_set_name, algorithm_name, algorithm_config, fold, num_samples=None, seed=None):
    configurations = retrieve_configurations(algorithm_name=algorithm_name, num_samples=num_samples, seed=seed)

    predict_log_file = (f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
                        f"config_{algorithm_config}/fold_{fold}/predict_log.json")
    with open(predict_log_file, "r") as file:
        predict_log = json.load(file)
    pipeline_file = predict_log["pipeline_file"]

    test = lenskit_load_transform(data_set_name, fold, "test")

    evaluate_log_dict = {
        "pipeline_file": pipeline_file,
        "data_set_name": data_set_name,
        "algorithm_name": algorithm_name,
        "algorithm_config_index": algorithm_config,
        "algorithm_configuration": configurations[algorithm_config],
        "fold": fold
    }


    with open(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
              f"config_{algorithm_config}/fold_{fold}/predictions.json", "r") as file:
        top_k_dict = json.load(file)

    top_k_dict = {int(k): v[0] for k, v in top_k_dict.items()}

    k_options = [1, 3, 5, 10, 20]

    start_evaluation = time.time()

    ndcg_per_user_per_k = ndcg(top_k_dict, k_options, test, "user", "item")
    hr_per_user_per_k = hr(top_k_dict, k_options, test, "user", "item")
    recall_per_user_per_k = recall(top_k_dict, k_options, test, "user", "item")
    end_evaluation = time.time()

    evaluate_log_dict["evaluation_time"] = end_evaluation - start_evaluation

    for k in k_options:
        score = sum(ndcg_per_user_per_k[k]) / len(ndcg_per_user_per_k[k])
        print(f"NDCG@{k}: {score}")
        evaluate_log_dict[f"NDCG@{k}"] = score
        score = sum(hr_per_user_per_k[k]) / len(hr_per_user_per_k[k])
        print(f"HR@{k}: {score}")
        evaluate_log_dict[f"HR@{k}"] = score
        score = sum(recall_per_user_per_k[k]) / len(recall_per_user_per_k[k])
        print(f"Recall@{k}: {score}")
        evaluate_log_dict[f"Recall@{k}"] = score


    with open(f"./data_sets/{data_set_name}/checkpoint_{algorithm_name}/"
              f"config_{algorithm_config}/fold_{fold}/evaluate_log.json", 'w') as file:
        json.dump(evaluate_log_dict, file, indent=4)
