# Welcome to LongDocURL!
Repository for the paper "LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating".

**Paper**: [arxiv](https://arxiv.org/pdf/2412.18424)

**Dataset Blog**: [https://longdocurl.github.io/](https://longdocurl.github.io/).

# About LongDocURL
The LongDocURL benchmark is specifically designed for assessing the ability of models in long document understanding.
We collect 2,325 high-quality question-answering pairs, covering 396 PDF-formatted documents and more than 33,000 pages, significantly outperforming existing benchmarks.
Our open dataset can be found at [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/). You can refer to [Dataset Website](https://longdocurl.github.io/) for more infomation.

# Evaluation

**1. (Optional) Download & Extract PDFs**

Download PDFs and qa file (.jsonl) from [LongDocURL](https://huggingface.co/datasets/dengchao/LongDocURL/). Run the following commands to extract PDFs into pngs and json files (by PyMuPDF).

```bash
bash utils/run_extract_ccpdf.sh
```

**2. Evaluating API Models**
```bash
bash scripts/eval_api_models.sh
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
