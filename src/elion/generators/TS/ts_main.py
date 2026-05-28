#!/usr/bin/env python

import importlib
import json
import sys
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd

from thompson_sampling_csv import ThompsonSamplerCSV
from ts_logger import get_logger


def read_input(json_filename: str) -> dict:
    """
    Read input parameters from a json file
    :param json_filename: input json file
    :return: a dictionary with the input parameters
    """
    input_data = None
    with open(json_filename, 'r') as ifs:
        input_data = json.load(ifs)
        module = importlib.import_module("evaluators")
        evaluator_class_name = input_data["evaluator_class_name"]
        class_ = getattr(module, evaluator_class_name)
        evaluator_arg = input_data["evaluator_arg"]
        evaluator = class_(evaluator_arg)
        input_data['evaluator_class'] = evaluator
    return input_data


def read_input_yml(yml_filename: str) -> dict:
    """
    Read TS input parameters from an Elion YAML config file.
    TS-specific parameters live under config['Generator']['TS'].
    The reward function config (config['Reward_function']) is used to
    build the ElionEstimatorEvaluator.
    :param yml_filename: path to the Elion input_TS.yml
    :return: input_data dict compatible with run_ts()
    """
    sys.path.append('../..')
    import input_reader
    config = input_reader.read_input_file(yml_filename)

    module = importlib.import_module("evaluators")
    class_ = getattr(module, "ElionEstimatorEvaluator")
    evaluator = class_(config['Reward_function'])

    ts_cfg = config['Generator']['TS']
    input_data = {
        'evaluator_class':      evaluator,
        'evaluator_class_name': 'ElionEstimatorEvaluator',
        'reaction_smarts':      ts_cfg['reaction_smarts'],
        'num_warmup_trials':    ts_cfg['num_warmup_trials'],
        'num_ts_iterations':    ts_cfg['num_ts_iterations'],
        'reagent_file_list':    ts_cfg['reagent_file_list'],
        'ts_mode':              ts_cfg['ts_mode'],
        'results_filename':     ts_cfg.get('results_filename'),
        'log_filename':         ts_cfg.get('log_filename'),
    }
    return input_data


def parse_input_dict(input_data: dict) -> None:
    """
    Parse the input dictionary and add the necessary information
    :param input_data:
    """
    module = importlib.import_module("evaluators")
    evaluator_class_name = input_data["evaluator_class_name"]
    class_ = getattr(module, evaluator_class_name)
    evaluator_arg = input_data["evaluator_arg"]
    evaluator = class_(evaluator_arg)
    input_data['evaluator_class'] = evaluator


def run_ts(input_dict: dict, hide_progress: bool = False) -> None:
    """
    Perform a Thompson sampling run
    :param hide_progress: hide the progress bar
    :param input_dict: dictionary with input parameters
    """
    evaluator = input_dict["evaluator_class"]
    reaction_smarts = input_dict["reaction_smarts"]
    num_ts_iterations = input_dict["num_ts_iterations"]
    reagent_file_list = input_dict["reagent_file_list"]
    num_warmup_trials = input_dict["num_warmup_trials"]
    result_filename = input_dict.get("results_filename")
    ts_mode = input_dict["ts_mode"]
    log_filename = input_dict.get("log_filename")
    logger = get_logger(__name__, filename=log_filename)
    ts = ThompsonSamplerCSV(mode=ts_mode, db_name="eXplore")
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)
    ts.read_reagents_csv(reagent_file_list=reagent_file_list, num_to_select=None)
    ts.set_reaction(reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    ts.warm_up(num_warmup_trials=num_warmup_trials)
    # run the search with TS
    out_list = ts.search(num_cycles=num_ts_iterations)
    total_evaluations = evaluator.counter
    percent_searched = total_evaluations / ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    # write the results to disk
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
    if result_filename is not None:
        out_df.to_csv(result_filename, index=False)
        logger.info(f"Saved results to: {result_filename}")
    if not hide_progress:
        if ts_mode == "maximize":
            print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))
        else:
            print(out_df.sort_values("score", ascending=True).drop_duplicates(subset="SMILES").head(10))
    return out_df


def run_10_cycles():
    """ A testing function for the paper
    :return: None
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in range(0, 10):
        input_dict['results_filename'] = f"ts_result_{i:03d}.csv"
        run_ts(input_dict, hide_progress=False)


def compare_iterations():
    """ A testing function for the paper
    :return:
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in (2, 5, 10, 50, 100):
        input_dict["num_ts_iterations"] = i * 1000
        input_dict["results_filename"] = f"iteration_test_{i}K.csv"
        run_ts(input_dict)


def main():
    start = timer()
    json_filename = sys.argv[1]
    input_dict = read_input(json_filename)
    run_ts(input_dict)
    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()