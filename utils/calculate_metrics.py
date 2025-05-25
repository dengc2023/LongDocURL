import argparse
import json

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

from utils.utils_score_v3 import eval_score

# modify: allow multiple preds and return score dict
def calculate_accuracy(answers: list, annotations: list, answer_formats: list, multiple_preds: list = None):
    total_scores = 0.0
    score_list = []
    for pred_ans, annotation, answer_format, multiple_pred in zip(answers, annotations, answer_formats, multiple_preds):
        if pred_ans == "Fail to extract":
            score_v3 = 0.0
        elif not multiple_pred:
            score_v3 = eval_score(annotation, pred_ans, answer_format)
        else:
            score_v3 = max([eval_score(annotation, item, answer_format) for item in pred_ans])
        
        score_list.append(score_v3)
        total_scores += score_v3
    
    generalized_score = total_scores / len(answers)

    return generalized_score, score_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default="")
    
    args = parser.parse_args()

    with open(args.results_file, "r", encoding="utf-8") as rf:
        samples = [json.loads(_.strip()) for _ in rf.readlines()]
    
    for sample in samples:
        assert "pred" in sample

    answers = [_["pred"] for _ in samples]
    annotations = [_["answer"] for _ in samples]
    answer_formats = [_["answer_format"] for _ in samples]
    
    # modify
    multiple_preds = [True if "multiple_pred" in _ else False for _ in samples] # for modified_answer case
    # multiple_preds = [False for _ in samples] # for non-modified_answer case
    
    generalized_score, score_list = calculate_accuracy(answers, annotations, answer_formats, multiple_preds) # calculate on size of successful samples
    rectified_generalized_score = generalized_score * len(answers) / 2325 # calculate on size of 2325
    
    # Update the score field in samples with scores from score_list
    for sample, score in zip(samples, score_list):
        sample["score"] = score
    
    # Write the updated results to a new file in the same directory
    output_file = args.results_file.replace(".jsonl", "_scored.jsonl")
    with open(output_file, "w", encoding="utf-8") as wf:
        for sample in samples:
            wf.write(json.dumps(sample) + "\n")
    print(f"Scores updated and saved to: {output_file}")


    print("--------------------------------------")
    print("Avg. acc: {}".format(generalized_score))
    print("Rectified Avg. acc: {}".format(rectified_generalized_score))
