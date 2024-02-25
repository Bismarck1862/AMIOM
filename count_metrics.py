import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = "datasets/results/"
SAVE_DIR = "datasets/results/metrics/"


def count_new(outfile, type="sent"):
    file_list = os.listdir(BASE_DIR)
    name_list = [f.split(".")[0] for f in file_list if f.startswith("nlp_experiment") and not "test" in f]
    out_path =  f"{SAVE_DIR}{outfile}.jsonl"
    counted = []
    if os.path.exists(out_path):
        with open(out_path, "r") as fin:
            for line in fin.readlines():
                counted.append(json.loads(line)["filename"])

    to_count_related = [f for f in name_list if f not in counted and "related" in f]
    to_count_words = [f for f in name_list if f not in counted and "words" in f]
    to_count_sent = [f for f in name_list if f not in counted and not "words" in f and not "related" in f]
    if type == "related":
        print(f"Counting related metrics, {len(to_count_related)} files")
        for filename in tqdm(to_count_related):
            count_related_metrics(filename, outfile)
    elif type == "words":
        print(f"Counting words metrics, {len(to_count_words)} files")
        for filename in tqdm(to_count_words):
            count_words_metrics(filename, outfile)
    else:
        print(f"Counting sentence metrics, {len(to_count_sent)} files")
        for filename in tqdm(to_count_sent):
            count_sent_metrics(filename, outfile)

# AMIOM metric
def get_metric(similarities_same, similarities_other):
    sim_pointwise_difference = similarities_other - similarities_same
    accuracy = np.sum(sim_pointwise_difference > 0) / len(sim_pointwise_difference)
    sim_mean = np.mean(sim_pointwise_difference)

    sim_max = np.abs(np.max(sim_pointwise_difference))
    sim_min = np.min(sim_pointwise_difference)
    return (sim_mean * sim_max) / ((1 - sim_min)) * 1000, accuracy
            

def count_sent_metrics(filename, outfile):
    print(f"Counting metrics for: {filename}")
    file_path = f"{BASE_DIR}{filename}.jsonl"
    out_path = f"{SAVE_DIR}{outfile}.jsonl"
    lines = []
    with open(file_path, "r") as fin:
        for line in fin.readlines():
            lines.append(json.loads(line))
    print(len(lines))
    sims_same = [line["sims_same"] for line in lines]
    sims_other = [line["sims_other"] for line in lines]
    all_sims = sims_same + sims_other
    gl_min = min(all_sims)
    gl_max = max(all_sims)
    # min max normalization
    sims_same = [(i - gl_min) / (gl_max - gl_min) for i in sims_same]
    sims_other = [(i - gl_min) / (gl_max - gl_min) for i in sims_other]
    sims_same = np.array(sims_same)
    sims_other = np.array(sims_other)
    metric, acc = get_metric(sims_same, sims_other)
    metrics = {
        "filename": filename,
        "same_mean": np.mean(sims_same),
        "same_std": np.std(sims_same),
        "same_sim": 1 - np.mean(sims_same),
        "other_mean": np.mean(sims_other),
        "other_std": np.std(sims_other),
        "other_sim": 1 - np.mean(sims_other),
        "metric": metric,
        "accuracy": acc
    }

    with open(out_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def count_related_metrics(filename, outfile):
    print(f"Counting metrics for: {filename}")
    file_path = f"{BASE_DIR}{filename}.jsonl"
    out_path = f"{SAVE_DIR}{outfile}.jsonl"
    lines = []
    with open(file_path, "r") as fin:
        for line in fin.readlines():
            lines.append(json.loads(line))
    print(len(lines))
    lines = [line for line in lines if line["sims_other"] > 0 and line["sims_same"] > 0]
    sims_related = [line["sims_same"] for line in lines]
    sims_unrelated = [line["sims_other"] for line in lines]
    all_sims = sims_related + sims_unrelated
    gl_min = min(all_sims)
    gl_max = max(all_sims)
    # min max normalization
    sims_related = [(i - gl_min) / (gl_max - gl_min) for i in sims_related]
    sims_unrelated = [(i - gl_min) / (gl_max - gl_min) for i in sims_unrelated]
    sims_related = np.array(sims_related)
    sims_unrelated = np.array(sims_unrelated)
    metric, acc = get_metric(sims_related, sims_unrelated)
    metrics = {
        "filename": filename,
        "related_sim": np.mean(sims_related),
        "related_std": np.std(sims_related),
        "related_mean": 1 - np.mean(sims_related),
        "unrelated_sim": np.mean(sims_unrelated),
        "unrelated_std": np.std(sims_unrelated),
        "unrelated_mean": 1 - np.mean(sims_unrelated),
        "metric": metric,
        "accuracy": acc
    }

    with open(out_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

def count_words_metrics(filename, outfile):
    print(f"Counting metrics for: {filename}")
    file_path = f"{BASE_DIR}{filename}.jsonl"
    out_path =  f"{SAVE_DIR}{outfile}.jsonl"
    lines = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            lines.append(json.loads(line))
    max_len = 20
    sims_same = [[] for _ in range(max_len)]
    sims_other = [[] for _ in range(max_len)]
    for line in lines:
        length = min(len(line["sims_same"]), len(line["sims_other"]))
        for i in range(length):
            if i < max_len:
                sims_same[i].append(line["sims_same"][i])
                sims_other[i].append(line["sims_other"][i])
    
    all_sims = []
    for i in range(max_len):
        all_sims += sims_same[i] + sims_other[i]
    gl_min = min(all_sims)
    gl_max = max(all_sims)
    
    for i in range(max_len):
        sims_same[i] = [(x - gl_min) / (gl_max  - gl_min) for x in sims_same[i]]
        sims_other[i] = [(x - gl_min) / (gl_max  - gl_min) for x in sims_other[i]]

    metrics = {"filename": filename,
                "same_sim": [],
                "same_std": [],
                "same_mean": [],
                "other_sim": [],
                "other_std": [],
                "other_mean": [],
                "metric": [],
                "accuracy": []
                }
    for i in range(max_len):
        metrics["same_sim"].append(np.mean(sims_same[i]))
        metrics["same_std"].append(np.std(sims_same[i]))
        metrics["same_mean"].append(1 - np.mean(sims_same[i]))
        metrics["other_sim"].append(np.mean(sims_other[i]))
        metrics["other_std"].append(np.std(sims_other[i]))
        metrics["other_mean"].append(1 - np.mean(sims_other[i]))
        metric, acc = get_metric(np.array(sims_other[i]), np.array(sims_same[i]))
        metrics["metric"].append(metric)
        metrics["accuracy"].append(acc)


    with open(out_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

count_new("related_metrics", "related")
count_new("sent_metrics", "sent")
count_new("words_metrics", "words")