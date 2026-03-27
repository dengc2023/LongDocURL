import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

# just for debug
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
import multiprocessing
import datetime
from openai import OpenAI

from eval.utils_api import *
from utils.utils_score_v3 import *
from eval.model import Gemini15ProInferencer, GPT4oInferencer, QwenVLMaxInferencer, O1PreviewInferencer, QwenMaxInferencer, Qwen2VLInferencer
from pure_ocr_utils import *

import torch

system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"

# TODO
project_prefix = "/home/dataset-local/Projects/CodeLib/LongDocURL/"

config_file = os.path.join(project_prefix, "config/api_config.json")
extractor_prompt_path = os.path.join(project_prefix, "eval/prompt_for_answer_extraction.md")

with open(config_file, "r", encoding="utf-8") as rf:
    config = json.load(rf)
# client = OpenAI(api_key=config["gpt4o"]["access_key"], base_url=config["gpt4o"]["base_url"])
client = OpenAI(api_key=config["qwen"]["access_key"], base_url=config["qwen"]["base_url"])

model_name2inferencer = {
    "gpt4o": "GPT4oInferencer", 
    "gemini15_pro": "Gemini15ProInferencer", 
    "qwen_vl_max": "QwenVLMaxInferencer",
    "o1_preview": "O1PreviewInferencer", 
    "qwen_max": "QwenMaxInferencer", 
    "qwen2-vl-7b": "Qwen2VLInferencer",
    "qwen25-vl-7b": "Qwen2VLInferencer"
}

model_pool = {}

prompt_sign = True

def preprocess(input_datapath, output_datapath, image_prefix=None):
    dataset = read_jsonl_file(input_datapath)
    print("dataset cnt: ", len(dataset))

    if os.path.exists(output_datapath):
        output_dataset = read_jsonl_file(output_datapath)
        dataset = delete_generate_dataset(dataset, output_dataset)

    if image_prefix is not None:
        for _ in dataset:
            for i, image_path in enumerate(_["images"]):
                _["images"][i] = os.path.join(image_prefix, "/".join(image_path.split("/")[-2:]))

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
            # completion = client.chat.completions.create(model="gpt-4o-0513", messages=msgs, temperature=0.)
            completion = client.chat.completions.create(model="qwen-turbo-latest", messages=msgs, temperature=0.)
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

def eval_per_record(model_pool, task=None, gpu_id=None, inferencer=None):
    case, output_datapath, model_name = task

    if inferencer is None:
        inferencer = eval(model_name2inferencer[model_name])(model_name)

    question = case["question"]
    prompt = system_prompt + "Following is our question: \n" + f"<question>{question}</question>" + "\n"

    try:
        result = inferencer.infer(prompt, case["images"], device=f"cuda:{gpu_id}" if gpu_id is not None else "cpu", model_pool=model_pool)
    except Exception as e:
        print("error: ", e)
        result = None

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

    if result is not None:  # check if result is not None
        try: # not json serialable
            os.makedirs(os.path.dirname(output_datapath), exist_ok=True)
            with open(output_datapath, "a") as output_review_file:
                output_review_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            print("error: ", e)
    else:
        print("error")


def task_scheduler(task_queue, gpu_queue, progress_counter, active_tasks):
    gpu_id = gpu_queue.get()
    inferencer = None
    local_model_pool = {}  # Local pool for this process, NO Manager sharing for models
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
        except:
            break

        case, output_datapath, model_name = task
        if inferencer is None:
            inferencer = eval(model_name2inferencer[model_name])(model_name)

        # Register current doc_no
        doc_no = case.get("doc_no", "unknown")
        active_tasks[gpu_id] = doc_no

        eval_per_record(local_model_pool, task, gpu_id, inferencer=inferencer)

        progress_counter.value += 1

    if gpu_id in active_tasks:
        del active_tasks[gpu_id]
    gpu_queue.put(gpu_id)  # put GPU resources back into the queue


def evaluate(dataset, output_datapath, model_name="gpt4o", num_gpus=0):

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

    # Use parallel mode if multiple GPUs are available
    if num_gpus > 1:
        print(f"Running in multi-GPU parallel mode on {num_gpus} devices.")
        task_queue = multiprocessing.Queue()
        for args in args_list:
            task_queue.put(args)

        gpu_queue = multiprocessing.Queue()
        for gpu_id in range(num_gpus):
            gpu_queue.put(str(gpu_id))

        with multiprocessing.Manager() as manager:
            progress_counter = manager.Value('i', 0)
            active_tasks = manager.dict() # track what each GPU is doing
            processes = []

            for i in range(num_gpus):
                p = multiprocessing.Process(target=task_scheduler, args=(task_queue, gpu_queue, progress_counter, active_tasks))
                p.start()
                processes.append(p)

            # start the progress bar monitoring thread
            with tqdm(total=len(args_list), desc="Processing (Parallel)") as pbar:
                last_progress = 0
                while any(p.is_alive() for p in processes):
                    current_progress = progress_counter.value
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress

                    # Update progress bar description with active documents
                    if active_tasks:
                        active_docs = list(active_tasks.values())
                        pbar.set_postfix(docs=active_docs, refresh=True)

                    time.sleep(0.5)

                # make sure the final progress shows 100%
                pbar.n = len(args_list)
                pbar.refresh()

            for p in processes:
                p.join()
    else:
        global model_pool
        # Serial mode for 1 GPU or CPU
        mode_str = "GPU:0" if num_gpus == 1 else "CPU"
        print(f"Running in serial mode on {mode_str}.")
        inferencer = eval(model_name2inferencer[model_name])(model_name)
        gpu_id = "0" if num_gpus == 1 else None

        with tqdm(total=len(args_list), desc=f"Processing ({mode_str})") as pbar:
            for args in args_list:
                case = args[0]
                pbar.set_postfix(doc=case.get("doc_no", "unknown"))
                eval_per_record(model_pool, args, gpu_id=gpu_id, inferencer=inferencer)
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', type=str, default="LongDocURL_public.jsonl")
    parser.add_argument('--results_file', type=str, default="evaluation_results/open_lvlms/results_qwen2vl_7b.jsonl")
    parser.add_argument('--process_mode', type=str, default="serial") # No longer used for open models, kept for compatibility if needed
    # parser.add_argument('--input_format', type=str, default="e2e") # e2e/ocr
    parser.add_argument('--image_prefix', type=str, default="/data/data/LongDocURL/pdf_pngs/4000-4999")
    parser.add_argument('--model_name', type=str, default="qwen2-vl-7b") # gemini15_pro/claude35_sonnet/qwen_vl_max/gpt4o

    args = parser.parse_args()

    multiprocessing.set_start_method('spawn')

    # Auto-detect GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Total available GPUs: {num_gpus}")

    input_datapath = args.qa_file
    output_datapath = args.results_file

    # load data
    # dataset = preprocess(input_datapath, output_datapath)
    # if image paths are not modified in .jsonl file, add image prefix when executed
    dataset = preprocess(input_datapath, output_datapath, image_prefix=args.image_prefix)

    try_cnt = 2
    while try_cnt:
        try_cnt -= 1
        try:
            evaluate(dataset, output_datapath, model_name=args.model_name, num_gpus=num_gpus)
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting script...")
            time.sleep(1)

    acc, f1, = calculate_acc_and_f1(output_datapath)
    print("--------------------------------------")
    print("Avg acc: {}".format(acc))
    print("Avg f1: {}".format(f1))