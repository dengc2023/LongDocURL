import re
import json
from math import isclose
from collections import defaultdict


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls<=threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    s = s.replace(",", "")
    if s.endswith("kg"):
        s = s.rstrip("kg").strip()
    if s.endswith("mm"):
        s = s.rstrip("mm").strip()
    if s.endswith("m"):
        s = s.rstrip("m").strip()
    if s.endswith("meters"):
        s = s.rstrip("meters").strip()
    if s.endswith("acres"):
        s = s.rstrip("acres").strip()
    if s.endswith("minutes"):
        s = s.rstrip("minutes").strip()
    if s.endswith("mile"):
        s = s.rstrip("mile").strip()
    if s.endswith("miles"):
        s = s.rstrip("miles").strip()
    if s.endswith("million"):
        s = s.rstrip("million").strip()
    if s.endswith("thousand"):
        s = s.rstrip("thousand").strip()
    if s.endswith("billion"):
        s = s.rstrip("billion").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().lstrip("£").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except Exception as e:
        return False


def eval_score(gt, pred, answer_type):
    if answer_type=="Integer":
        try:
            gt = get_clean_string(str(gt))
            if len(re.findall(r"\d+,\s*\d+", gt, re.DOTALL)) > 0: # deal with Integer value formatted as "96,395"
                gt = "".join([_.strip() for _ in gt.split(",")])
            gt = int(gt)
        except:
            gt = gt
        try:
            pred = get_clean_string(str(pred))
            if len(re.findall(r"\d+,\s*\d+", pred, re.DOTALL)) > 0: # deal with Integer value formatted as "96,395"
                pred = "".join([_.strip() for _ in pred.split(",")])
            pred = int(pred)
        except:
            pred = ""
        score = (gt==pred)
    elif answer_type=="Float":
        gt = get_clean_string(str(gt))
        pred = get_clean_string(str(pred))
        
        if len(re.findall(r"\d+,\s*\d+", gt, re.DOTALL)) > 0: # deal with Integer value formatted as "96,395"
            gt = "".join([_.strip() for _ in gt.split(",")])
        try:
            gt = float(gt)
        except:
            gt = gt
        
        if len(re.findall(r"\d+,\s*\d+", pred, re.DOTALL)) > 0: # deal with Integer value formatted as "96,395"
            pred = "".join([_.strip() for _ in pred.split(",")])
        try:
            pred = float(pred)
        except:
            pred = str(pred)

        try:
            score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
        except:
            score = 0

    elif answer_type in ["String", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt==pred)
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            try:
                gt = eval(gt)
            except:
                gt = gt
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            try:
                pred = eval(pred)
            except:
                pred = pred
        if not isinstance(pred, list):
            pred = [pred]
        if isinstance(gt[0], dict):
            gt = ["-".join([str(value) for key,value in _.items()]) for _ in gt]
        if isinstance(pred[0], dict):
            pred = ["-".join([str(value) for key,value in _.items()]) for _ in pred]

        print(len(gt), len(pred))
        print(gt, pred)
        def cal_score_v3(gt, pred):
            gt = [get_clean_string(a) for a in gt]
            pred = [get_clean_string(a) for a in pred]
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score_v3 = ("-".join(gt)=="-".join(pred))
            else:
                greedy_scores = [max([anls_compute(str(gt_v), str(pred_v)) for pred_v in pred]) for gt_v in gt]
                score_v3 = sum(greedy_scores) / len(gt) * min(1, len(gt) / len(pred)) ** 0.5
            return score_v3

        score_v3 = cal_score_v3(gt, pred)

    score_v3 = score if answer_type in ["Integer", "Float", "String", "None"] else score_v3

    return float(score_v3)


def calculate_acc_and_f1(results_file):
    samples = [json.loads(_.strip()) for _ in open(results_file, "r", encoding="utf-8").readlines()]
    evaluated_samples = [sample for sample in samples if "score_v3" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    
    acc = sum([sample["score_v3"] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample["score_v3"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["answer"]!="Not answerable"])
        precision = sum([sample["score_v3"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["pred"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0
    
    return acc, f1


def show_results(samples, show_path=None):
    for sample in samples:
        sample["evidence_pages"] = eval(sample["evidence_pages"])
        sample["evidence_sources"] = eval(sample["evidence_sources"])
    
    with open(show_path, 'w') as f:
        acc, f1 = eval_acc_and_f1(samples)
        f.write("Overall Acc: {} | Question Number: {}\n".format(acc, len(samples)))
        f.write("Overall F1-score: {} | Question Number: {}\n".format(f1, len(samples)))
        f.write("-----------------------\n")

        #####################
        acc_single_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])==1])
        acc_multi_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        acc_neg, _ = eval_acc_and_f1([sample for sample in samples if sample["answer"]=="Not answerable"])

        f.write("Single-page | Accuracy: {} | Question Number: {}\n".format(
            acc_single_page, len([sample for sample in samples if len(sample["evidence_pages"])==1])
        ))
        f.write("Cross-page | Accuracy: {} | Question Number: {}\n".format(
            acc_multi_page, len([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        ))
        f.write("Unanswerable | Accuracy: {} | Question Number: {}\n".format(
            acc_neg, len([sample for sample in samples if sample["answer"]=="Not answerable"])
        ))
        f.write("-----------------------\n")

        #####################
        source_sample_dict, document_type_dict = defaultdict(list), defaultdict(list)
        for sample in samples:
            for answer_source in sample["evidence_sources"]:
                source_sample_dict[answer_source].append(sample)
            document_type_dict[sample["doc_type"]].append(sample)
        for type, sub_samples in source_sample_dict.items():
            f.write(
                "Evidence Sources: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

        f.write("-----------------------\n")
        for type, sub_samples in document_type_dict.items():
            f.write(
                "Document Type: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )
