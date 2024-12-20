# # TODO:
# import sys
# sys.path.append("/root_dir/LongDocURL/")

import argparse
import os
from io import BytesIO

import oss2
import base64
import json
from tqdm import tqdm
import requests
import re
import time
from multiprocessing import Pool
import datetime
from openai import OpenAI

from eval.utils_api import *
from utils.utils_score_v3 import *

system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"

# TODO
project_prefix = "/root_dir/LongDocURL/"
config_file = os.path.join(project_prefix, "config/api_config.json")
extractor_prompt_path = os.path.join(project_prefix, "eval/prompt_for_answer_extraction.md")

with open(config_file, "r", encoding="utf-8") as rf:
    config = json.load(rf)
client = OpenAI(api_key=config["gpt4o"]["access_key"], base_url=config["gpt4o"]["base_url"])


def call_llm(prompt, urls, temperature=0.1, seed=42, max_tokens=4096):
    msgs = get_msg_format(prompt, urls)
    response = None
    max_try = 6
    while response is None and max_try > 0:
        try:
            completion = client.chat.completions.create(model="gpt-4o-0513", messages=msgs, temperature=0.)
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"error with {e}, response = {response}")
            max_try -= 1
            response = None

    return response


def delete_generated_dataset(records, output_datapath):
    finished_sample_ids = set([json.loads(_.strip())["question_id"] for _ in open(output_datapath, "r", encoding="utf-8").readlines()]) if os.path.exists(output_datapath) else set()
    return [_ for _ in records if _["question_id"] not in finished_sample_ids]


def extract_per_record(args):
    case, result, output_datapath = args
    question = case["question"]
    print(case["question_id"])

    # extract concise answer
    with open(extractor_prompt_path) as f:
        extractor_prompt = f.read()
    prompt = system_prompt + extractor_prompt + "\nQuestion: " + question + "\nAnalysis: " + result
    extractor_result = call_llm(prompt, None)
    try:
        import re
        concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", extractor_result, re.DOTALL)[0]
        answer_format = re.findall(r"<answer_format>(.*?)</answer_format>", extractor_result, re.DOTALL)[0]
    except:
        concise_answer = "Fail to extract"
        answer_format = "None"

    # calculate scores
    try:
        # pred_ans = eval(concise_answer)
        pred_ans = eval(concise_answer) if not isinstance(eval(concise_answer), set) else list(eval(concise_answer))
    except:
        pred_ans = concise_answer
    if pred_ans == "Fail to extract":
        score_v3 = 0.0
    else:
        score_v3 = eval_score(case["answer"], pred_ans, case["answer_format"])

    case["pred"] = pred_ans
    case["score_v3"] = score_v3

    print("\n\n")
    print("Question: {}".format(case["question"]))
    print("Response: {}".format(case["pred"]))
    print("Gt: {}\tPred: {}\tScore_v3: {}".format(case["answer"], case["pred"], case["score_v3"]))

    try:
        with open(output_datapath, "a") as output_review_file:
            output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
    except Exception as e:
        print("error: ", e)
        print("error: ", case["question_id"])



def extract_answers(records, output_datapath):

    records = delete_generated_dataset(records, output_datapath)
    args_list = []
    for record in records:
        args_list.append((record, record["detailed_response"], output_datapath))

    with Pool(processes=1) as pool:
        list(tqdm(pool.imap(extract_per_record, args_list), total=len(args_list)))


# def run_test():
#     # case = {}
#     question = case["question"]
#     result = case["detailed_response"]
#     output_datapath = ""
#     args = (case, question, result, output_datapath)
#     extract_per_record(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, default="./evaluation_results/api_models/results_detailed.jsonl")
    parser.add_argument("--results_file", type=str, default="./evaluation_results/api_models/results_extracted.jsonl")
    args = parser.parse_args()

    with open(args.qa_file, "r", encoding="utf-8") as rf:
        records = [json.loads(_.strip()) for i, _ in enumerate(rf.readlines())]

    extract_answers(records, args.results_file)

    # run_test()


