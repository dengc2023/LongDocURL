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
from model import Gemini15ProInferencer, GPT4oInferencer, QwenVLMaxInferencer, O1PreviewInferencer, QwenMaxInferencer
from pure_ocr_utils import *

system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"

# TODO
project_prefix = "/root_dir/LongDocURL/"
config_file = os.path.join(project_prefix, "config/api_config.json")
extractor_prompt_path = os.path.join(project_prefix, "eval/prompt_for_answer_extraction.md")

with open(config_file, "r", encoding="utf-8") as rf:
    config = json.load(rf)
client = OpenAI(api_key=config["gpt4o"]["access_key"], base_url=config["gpt4o"]["base_url"])

model_name2inferencer = {"gpt4o": "GPT4oInferencer", "gemini15_pro": "Gemini15ProInferencer", "qwen_vl_max": "QwenVLMaxInferencer", \
    "o1_preview": "O1PreviewInferencer", "qwen_max": "QwenMaxInferencer"}

prompt_sign = True

def preprocess(input_datapath, output_datapath):
    dataset = read_jsonl_file(input_datapath)
    print("dataset cnt: ", len(dataset))

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    print("dataset cnt need to do: ", len(dataset))

    return dataset

def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as jsonl_file:
        for i, line in enumerate(jsonl_file):
            data_dict = json.loads(line.strip())
            if 'question_id' not in data_dict:
                data_dict['question_id'] = i
            data.append(data_dict)
    return data

def call_llm(prompt, urls, temperature=0.1, seed=42, max_tokens=4096):
    msgs = get_msg_format(prompt, urls)
    response = None
    max_try = 2
    while response is None and max_try > 0:
        try:
            # TODO
            completion = client.chat.completions.create(model="gpt-4o-0513", messages=msgs, temperature=0.)
            # completion = client.chat.completions.create(model="gpt-4o", messages=msgs, temperature=0.)
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"error with {e}, response = {response}")
            max_try -= 1
            response = None

    return response

def delete_generate_dataset(dataset, output_dataset):
    finished_question_id_set = set([sample['question_id'] for sample in output_dataset])
    unfinished_dataset = [sample for sample in dataset if sample['question_id'] not in finished_question_id_set]
    return unfinished_dataset

def eval_per_record(args):
    print("--------------------------------------")
    case, output_datapath, model_name = args

    inferencer = eval(model_name2inferencer[model_name])()

    question = case["question"]
    prompt = system_prompt + "Following is our question: \n" + f"<question>{question}</question>" + "\n"

    result = inferencer.infer(prompt, case["images"])

    if result is None:
        return

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
        
    case["detailed_response"] = result
    case["pred"] = pred_ans
    case["score_v3"] = score_v3

    print("\n\n")
    print("Question: {}".format(case["question"]))
    print("Response: {}".format(case["pred"]))
    
    print("Gt: {}\tPred: {}\tScore_v3: {}".format(case["answer"], case["pred"], case["score_v3"]))

    if result is not None:  # Check if result is not None
        try: # not json serialable
            with open(output_datapath, "a") as output_review_file:
                output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            print("error: ", e)
    else:
        print("error")


def evaluate(dataset, output_datapath, model_name="gpt4o", process_mode="serial", extra_infos=None):

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    print("dataset cnt: ", len(dataset))
    if not len(dataset):
        return

    args_list = []
    for case in dataset:
        args_list.append((case, output_datapath, model_name))

    start_time = datetime.datetime.now()
    print("job start time:", start_time)

    if process_mode == "serial":
        for args in args_list:
            eval_per_record(args)
    elif process_mode == "parallel":
        with Pool(processes=8) as pool:  # You can adjust the number of processes as needed
            list(tqdm(pool.imap(eval_per_record, args_list), total=len(args_list)))
    else:
        print("process mode error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', type=str, default="")
    parser.add_argument('--results_file', type=str, default="") 
    parser.add_argument('--process_mode', type=str, default="parallel") # serial/parallel
    # parser.add_argument('--input_format', type=str, default="e2e") # e2e/ocr
    parser.add_argument('--model_name', type=str, default="gpt4o") # gemini15_pro/gemini15_pro_flash/claude35_sonnet/claude3_haiku/gpt4o

    args = parser.parse_args()

    input_datapath = args.qa_file
    output_datapath = args.results_file

    # load data
    dataset = preprocess(input_datapath, output_datapath)

    try_cnt = 2
    while try_cnt:
        try_cnt -= 1
        try:
            evaluate(dataset, output_datapath, model_name=args.model_name, process_mode=args.process_mode)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting script...")
            time.sleep(1)

    acc, f1, = calculate_acc_and_f1(output_datapath)
    print("--------------------------------------")
    print("Avg acc: {}".format(acc))
    print("Avg f1: {}".format(f1))