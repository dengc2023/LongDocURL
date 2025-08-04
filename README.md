# Welcome to LongDocURL!
Repository for the ACL 2025 main paper "LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating".

**Resources**: [Paper](https://arxiv.org/pdf/2412.18424v3) • [Project Page](https://longdocurl.github.io/) • [Dataset](https://huggingface.co/datasets/dengchao/LongDocURL/)

# About LongDocURL
The LongDocURL benchmark is specifically designed for assessing the ability of models in long document understanding.
We collect 2,325 high-quality question-answering pairs, covering 396 PDF-formatted documents and more than 33,000 pages, significantly outperforming existing benchmarks.
Our open dataset can be found at [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/). You can refer to [Blog Website](https://longdocurl.github.io/) for more infomation.

# Environment Setup (Qwen2-VL Series)

## Prerequisites

- Python == 3.10
- CUDA == 12.2

## Installation

1. Create and activate a virtual environment:
```bash
conda create -n longdocurl python=3.10 -y  # Linux/MacOS
conda activate longdocurl # or source activate longdocurl
cd /path/to/LongDocURL
```

2. Install required packages:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. For flash-attn2 installation issues:
    - Download the appropriate wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
    - We used this version: [flash_attn-2.6.2+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
    - Install locally:

```bash
pip install path/to/downloaded/flash_attn-2.6.2+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

# Evaluation

**1. Download & Extract PDFs** (If you have already downloaded the PNG package, please skip this step)

Download PDFs and qa file (.jsonl) from [HuggingFace](https://huggingface.co/datasets/dengchao/LongDocURL/). Run the following commands to extract PDFs into pngs and json files (by PyMuPDF).

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

**2. Configurations**
- `api_key`: update `config/api_config.json` with your API key. For Alibaba Cloud models, you can apply for free quota at [Bailian Console](https://bailian.console.aliyun.com/console?tab=model#/model-market). The configuration should include both `api_key` and `base_url`.
- `qa_jsonl`: update `data/LongDocURL.jsonl`. Note that this file only contains a small sample of QA pairs. The complete dataset can be downloaded from [HuggingFace](https://huggingface.co/datasets/dengchao/LongDocURL/).
- `api_models`: by default, we use `gpt4o-2024-05-13` for extracting short answers when evaluating API models. For open-source models, we use `qwen-turbo` for short answer extraction. If you use our code to evaluate proprietary models, please check and modify `eval/api_models/model.py`.

**3. Evaluating**

For api models:

```bash
bash scripts/eval_api_models.sh
```

For open-source models (currently only support qwen2-vl and qwen2.5-vl series):

```bash
bash scripts/eval_open_lvlms.sh
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

#  🏆 Leaderboard 🏆

| Model                     | Size   | Understanding  | Reasoning   | Locating   | Total |
|---------------------------|--------|----------------|-------------|------------|-------|
|	GPT-4o-24-05-13 🥇       | -      | 68.6           | 59.9        | 59.6       | 64.5  |
| Gemini-1.5-Pro 🥈        | -      | 55.7           | 43.4        | 46.4       | 50.9  |
| Qwen-VL-Max 🥉           | -      | 58.8           | 43.9        | 36.0       | 49.5  |
| Qwen2-VL                  | 7B     | 36.9           | 24.8        | 22.6       | 30.6  |
| LLaVA-OneVision-Chat      | 7B     | 30.5           | 19.0        | 18.7       | 25.0  |
| LLaVA-Next-Interleave-DPO | 7B     | 21.6           | 13.9        | 7.6        | 16.2  |
| Llama-3.2                 | 11B    | 12.9           | 9.4         | 2.7        | 9.2   |
