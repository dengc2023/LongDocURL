import argparse
import json

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

from utils.utils_score_v3 import eval_score

def calculate_accuracy_fine_grained(samples, score_dict):
    for sample in samples:
        pred_ans, annotation, answer_format, multiple_pred = sample["pred"], sample["answer"], sample["answer_format"], True if "multiple_pred" in sample else False
        if pred_ans == "Fail to extract":
            score_v3 = 0.0
        elif not multiple_pred:
            score_v3 = eval_score(annotation, pred_ans, answer_format)
        else:
            score_v3 = max([eval_score(annotation, item, answer_format) for item in pred_ans])
        sample["score_v3"] = score_v3
        
    # Main_Task
    for sample in samples:
        score_dict["Main_Task"][sample["task_tag"]] += sample["score_v3"]
    
    # Element_Type
    for sample in samples:
        for evidence_source in sample["evidence_sources"]:
            if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                score_dict["Element_Type"][evidence_source] += sample["score_v3"]

    # Evidence_Pages
    for sample in samples:
        if len(sample["evidence_pages"]) > 1:
            score_dict["Evidence_Pages"]["Multi_Page"] += sample["score_v3"]
        elif len(sample["evidence_pages"]) == 1:
            score_dict["Evidence_Pages"]["Single_Page"] += sample["score_v3"]

    # Num_of_Element_Types
    for sample in samples:
        if len(sample["evidence_sources"]) > 1:
            score_dict["Num_of_Element_Types"]["Cross_Element"] += sample["score_v3"]

    # Fine_Grained
    for sample in samples:
        sub_score_dict = score_dict["Fine_Grained"][sample["task_tag"]]
        if sample["task_tag"] in ["Understanding", "Reasoning"]:
            if len(sample["evidence_pages"]) > 1:
                sub_sub_score_dict = sub_score_dict["Multi_Page"]
            elif len(sample["evidence_pages"]) == 1:
                sub_sub_score_dict = sub_score_dict["Single_Page"]

            for evidence_source in sample["evidence_sources"]:
                if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                    sub_sub_score_dict[evidence_source] += sample["score_v3"]

            if len(sample["evidence_pages"]) > 1:
                sub_score_dict["Multi_Page"] = sub_sub_score_dict
            elif len(sample["evidence_pages"]) == 1:
                sub_score_dict["Single_Page"] = sub_sub_score_dict

        elif sample["task_tag"] in ["Locating"]:
            sub_sub_score_dict = sub_score_dict["Cross_Element"]
            if sample["question_type"] == "topic2title":
                sub_sub_score_dict["Cross_Title"] += sample["score_v3"]
            elif sample["question_type"] == "summary2title":
                sub_sub_score_dict["Para_Title"] += sample["score_v3"]
            elif sample["question_type"] == "summary2tab":
                sub_sub_score_dict["Cross_Table"] += sample["score_v3"]
            elif sample["question_type"] == "extract_fig2tab":
                sub_sub_score_dict["Figure_Table"] += sample["score_v3"]
            
            sub_score_dict["Cross_Element"] = sub_sub_score_dict
        
        score_dict["Fine_Grained"][sample["task_tag"]] = sub_score_dict


    return score_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default="evaluation_results/api_models/results_gemini15_pro.jsonl")
    parser.add_argument('--score_sample_file', type=str, default="evaluation_results/scores_sample_fine_grained.json")
    
    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as rf:
        samples = [json.loads(_.strip()) for _ in rf.readlines()]

    with open(args.score_sample_file, "r", encoding="utf-8") as rf:
        _ = json.load(rf)
        score_dict, sample_cnt_dict = _["scores"], _["sample_cnt"]
    
    for sample in samples:
        assert "pred" in sample

    score_dict = calculate_accuracy_fine_grained(samples, score_dict)

    def generalize_score_dict(score_dict, sample_cnt_dict):
        for key, value in score_dict.items():
            if isinstance(value, dict):
                generalize_score_dict(value, sample_cnt_dict[key])
                score_dict[key] = value
            else:
                score_dict[key] /= sample_cnt_dict[key]

    generalize_score_dict(score_dict, sample_cnt_dict)

    print("--------------------------------------------------------------")
    print(score_dict)
