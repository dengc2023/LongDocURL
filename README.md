# Welcome to LongDocURL!
Repository for the paper "LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating".

**Paper**: [paper](https://arxiv.org/pdf/2412.18424)

**Blog Website**: [longdocurl. github. io](https://longdocurl.github.io/)

**Dataset**: [dataset](https://huggingface.co/datasets/dengchao/LongDocURL/)

## About LongDocURL
The LongDocURL benchmark is specifically designed for assessing the ability of models in long document understanding.
We collect 2,325 high-quality question-answering pairs, covering 396 PDF-formatted documents and more than 33,000 pages, significantly outperforming existing benchmarks.
Our open dataset can be found at [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/). You can refer to [Blog Website](https://longdocurl.github.io/) for more infomation.

## Evaluation

**1. (Optional) Download & Extract PDFs**

Download PDFs and qa file (.jsonl) from [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/). Run the following commands to extract PDFs into pngs and json files (by PyMuPDF).

```bash
bash utils/run_extract_ccpdf.sh
```

Images will be organized in following ways:
```markdown
├── 4000
│   └── 4000001.png
└── 4001
    ├── 4001001.png
    └── 4001002.png
```

**2. Other Configurations**
- `api_key`: update `config/api_config.json`, used to extract short answer from detailed response.
- `qa_jsonl`: update `data/LongDocURL.jsonl`, downloaded from [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/).
- `api_models`: default `gpt4o-2024-05-13` for extracting short answer. if use our codes to evaluate proprietary models, please check and modify `eval/api_models/model.py`.

**3. Evaluating API Models**
```bash
bash scripts/eval_api_models.sh
```

Options to note:
- `process_mode`: default `serial`. Set `parallel` if parallel execution is needed. Default number of parallel processes is 8.
- `image_prefix`: default `None`. Add image prefix when needed in order to get proper image paths.
- `model_name`: the model abbreviation is mapped to the actual model class defined in `eval/api_models/model.py`,

**4. Claculate Metrics**

To calculate the final generalized accuracy:
```bash
bash scripts/calculate_metrics.sh
```
To calculate generalized accuracy in a more fine-grained way like `evaluation_results/scores_sample_fine_grained.json`:
```bash
bash scripts/calculate_metrics_fine_grained.sh
```

##  🏆 Leaderboard 🏆

| Model                     | Size   | Understanding  | Reasoning   | Locating   | Total |
|---------------------------|--------|----------------|-------------|------------|-------|
|	GPT-4o-24-05-13 🥇       | -      | 68.6           | 59.9        | 59.6       | 64.5  |
| Gemini-1.5-Pro 🥈        | -      | 55.7           | 43.4        | 46.4       | 50.9  |
| Qwen-VL-Max 🥉           | -      | 58.8           | 43.9        | 36.0       | 49.5  |
| Qwen2-VL                  | 7B     | 36.9           | 24.8        | 22.6       | 30.6  |
| LLaVA-OneVision-Chat      | 7B     | 30.5           | 19.0        | 18.7       | 25.0  |
| LLaVA-Next-Interleave-DPO | 7B     | 21.6           | 13.9        | 7.6        | 16.2  |
| Llama-3.2                 | 11B    | 12.9           | 9.4         | 2.7        | 9.2   |


## Citation

```
@article{chao-etal-2024-longdocurl,
  author       = {Chao Deng and
                  Jiale Yuan and
                  Pi Bu and
                  Peijie Wang and
                  Zhong{-}Zhi Li and
                  Jian Xu and
                  Xiao{-}Hui Li and
                  Yuan Gao and
                  Jun Song and
                  Bo Zheng and
                  Cheng{-}Lin Liu},
  title        = {LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating
                  Understanding, Reasoning, and Locating},
  journal      = {CoRR},
  volume       = {abs/2412.18424},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2412.18424},
  doi          = {10.48550/ARXIV.2412.18424},
  eprinttype    = {arXiv},
  eprint       = {2412.18424},
  timestamp    = {Sat, 25 Jan 2025 12:51:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2412-18424.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Please kindly cite our paper if this paper and the codes are helpful.
