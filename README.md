# PII Entity Recognition for Noisy STT Transcripts – Final Submission

**Model**: TinyBERT 4-layer + special tokens (`dot`, `at`, `oh`, `zero`, `dash`)  
**Training data**: 800 synthetic noisy STT examples (spoken digits, "at", "dot", "oh", etc.)  
**Dev data**: 150 synthetic noisy examples  

### Final Performance (on my 150-example noisy dev set)
- **PII Precision**: **82.8%** (strong — well above 80%)  
- PII Recall: 92.4%  
- p95 CPU latency (batch=1): **16.79 ms** (under 20 ms limit)  

### Files in this repo
- `src/` → all code (unchanged except tokenizer special tokens)  
- `data_new/` → 800 train + 150 dev noisy examples  
- `dev_pred.json` → predictions on dev set  

### Final Model + Predictions (~52 MB zip) – Google Drive
Download here: https://drive.google.com/file/d/1PEmoUaxH06zYzO81Imcm6QlFBsSwn8Bp/view?usp=drive_link

**How to use** (for evaluator):
1. Download and unzip → creates folder `out_final/`
2. Place it in repo root
3. Run:
```bash
python src/predict.py --model_dir out_final --input data_new/dev.jsonl
python src/eval_span_f1.py --gold data_new/dev.jsonl --pred out_final/dev_pred.json
python src/measure_latency.py --model_dir out_final --device cpu --runs 100
