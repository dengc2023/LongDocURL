# model_name ["qwen2-vl-7b", "qwen25-vl-7b"]
CUDA_VISIBLE_DEVICES=4,5 python eval/eval_open_lvlms.py \
    --qa_file data/LongDocURL.jsonl \
    --results_file evaluation_results/open_lvlms/results_qwen2vl_7b.jsonl \
    --process_mode serial \
    --image_prefix data/pdf_pngs/4000-4999 \
    --model_name qwen2-vl-7b \
    --devices 0,1