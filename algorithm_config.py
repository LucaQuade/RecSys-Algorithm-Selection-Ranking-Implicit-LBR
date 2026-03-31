import itertools
import math
import operator
import random
import hashlib
from functools import reduce


def retrieve_configurations(algorithm_name, num_samples=None, seed=None):
    configuration_space = {}
    if algorithm_name == "Pop":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "ItemKNN":
        configuration_space["epochs"] = [1]
        configuration_space["k"] = [10, 50, 100, 200, 250, 300, 400, 500, 1000, 1500, 2000, 2500]
        configuration_space["shrink"] = [0.0, 1.0]
    elif algorithm_name == "BPR":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "NeuMF":
        configuration_space["mf_embedding_size"] = [32, 64]
        configuration_space["mlp_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        configuration_space["dropout_prob"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif algorithm_name == "DMF":
        configuration_space["user_embedding_size"] = [64]
        configuration_space["item_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "LightGCN":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        configuration_space["n_layers"] = [1, 2, 3, 4]
        configuration_space["reg_weight"] = [1e-05, 1e-04, 1e-03, 1e-02]
    elif algorithm_name == "MultiVAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "MultiDAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "MacridVAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        configuration_space["kafc"] = [3, 5, 10, 20]
    elif algorithm_name == "LINE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        configuration_space["second_order_loss_weight"] = [0.3, 0.6, 1]
    elif algorithm_name == "CDAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "ENMF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        configuration_space["dropout_prob"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        configuration_space["negative_weight"] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    elif algorithm_name == "RecVAE":
        configuration_space["latent_dimension"] = [64, 100, 128, 150, 200, 256, 300, 400, 512]
        configuration_space["learning_rate"] = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    elif algorithm_name == "EASE":
        configuration_space["epochs"] = [1]
        configuration_space["reg_weight"] = [1.0, 10.0, 100.0, 250.0, 500.0, 1000.0]
    elif algorithm_name == "NCEPLRec":
        configuration_space["epochs"] = [1]
        configuration_space["rank"] = [100, 200, 450]
        configuration_space["beta"] = [0.8, 1.0, 1.3]
        configuration_space["reg_weight"] = [1e-04, 1e-02, 1e2, 15000]
    elif algorithm_name == "SimpleX":
        configuration_space["embedding_size"] = [64]
        configuration_space["gamma"] = [0.3, 0.5, 0.7]
        configuration_space["margin"] = [0.0, 0.5, 0.9]
        configuration_space["negative_weight"] = [0, 10, 50]
    elif algorithm_name == "Random":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "PopScore":
        configuration_space["score"] = ["quantile", "rank", "count"]
    elif algorithm_name == "ItemItem":
        configuration_space["max_nbrs"] = [5, 10, 20, 50, 100]
        configuration_space["min_nbrs"] = [1, 2, 5]
        configuration_space["min_sim"] = [1e-6, 1e-3]
    elif algorithm_name == "UserUser":
        configuration_space["max_nbrs"] = [10, 20, 50, 100]
        configuration_space["min_nbrs"] = [1, 5, 10]
        configuration_space["min_sim"] = [1e-6, 1e-3]
    elif algorithm_name == "ImplicitMF":
        configuration_space["embedding_size"] = [50, 75]
        configuration_space["epochs"] = [100, 150, 200]
        configuration_space["regularization"] = [0.01, 0.05, 0.1]
        configuration_space["weight"] = [100, 200, 500]
    elif algorithm_name == "SVD":
        configuration_space["num_components"] = [10, 20, 50, 100, 200]
    elif algorithm_name == "NMF":
        configuration_space["num_components"] = [10, 20, 50, 100, 200]
        configuration_space["alpha"] = [0.0, 0.0001, 0.001, 0.01, 0.1]
    elif algorithm_name == "ItemKNNRP":
        configuration_space["K"] = [50, 100, 200, 500]
        configuration_space["similarity"] = ["cosine", "conditional_probability"]

    if num_samples is None:
        num_samples = 2
    if seed is None:
        seed = 0

    #Generate seed
    seed_string = f"{algorithm_name}_{seed}"
    combined_seed = int(hashlib.md5(seed_string.encode("utf-8")).hexdigest(), 16)
    rnd = random.Random(combined_seed)

    keys = list(configuration_space.keys())
    values = list(configuration_space.values())

    max_samples = min(num_samples, reduce(operator.mul, (len(v) for v in configuration_space.values()), 1))

    #Randomly generate unique configurations
    samples = set()
    while len(samples) < max_samples:
        config = tuple((k, rnd.choice(v)) for k, v in zip(keys, values))
        samples.add(config)

    experiments = [dict(t) for t in sorted(samples)]
    return experiments
