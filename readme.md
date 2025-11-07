# TASU: Text-only Alignment for Speech Understanding

TASU (Text-only Alignment for Speech Understanding) is a newly proposed training paradigm for **Speech Large Language Models (Speech LLMs)**, with a primary focus on **semantic speech understanding**.  
This repository contains the **core implementation** of the TASU algorithm.

---

## ⚙️ Environment Setup

TASU is mainly developed and tested on **Huawei Ascend (910B) NPU clusters**, but can also be adapted to **NVIDIA GPU** environments.

### 1. Running on Huawei Ascend 910B (NPU)

We primarily run TASU on Huawei Ascend 910B clusters.  You can use the dockerfile like:
```bash
# Build image from Dockerfile in the current directory
docker build -t tasu:latest .
``` 

## 2. Running on NVIDIA GPUs

For NVIDIA GPU users, you can use the requirements.txt by:
```bash
pip install -r requirements.txt
```

- When adapting scripts or configs from the NPU version, replace a small number of `.npu` usages with `.gpu` (e.g., device tags or backend names).

> Full GPU-specific support is under active development, and we plan to provide more detailed GPU setup and scripts soon.

---

## 🚀 Getting Started

Once your environment is ready (either Ascend NPU or NVIDIA GPU):

1. Clone this repository:
   ```bash
   git clone https://github.com/PigeonDan1/ps-slm.git
   cd ps-slm
   cd Multitask
   ``` 
2. Prepare the dataset in format.
   
   Each sample is **one valid JSON line (JSON Lines)**. Field names and constraints:
   
   | Field  | Type   | Required | Description |
   |--------|--------|----------|-------------|
   | key    | string | ✔        | Globally unique ID, no `/` or spaces |
   | task   | string | ✔        | Task code (ASR, EN2ZH, etc.) |
   | target | string | ✔        | Text that the model must produce (label / decoding target) |
   | path   | string | ✔        | Audio location, 2 protocols supported, see below |
   | GT     | string | ✘(✔)     | Audio GT for text-simulation CTC posterior |

   Audio format support:
   
   | Protocol | Example Path | Reading Hint |
   |----------|--------------|--------------|
   | plain wav | `/xxx/common_voice_en_19641841.wav` | direct `soundfile.read` |
   | ark offset | `/xxx/data_wav.1.ark:246511401` | binary `seek(offset)` |

   Data examples:
   ```bash
   {"key": "common_voice_en_211671", "task": "ASR", "target": "That is a weird phrase.", "path": "/data/audio/dev/common_voice_en_211671.wav"}
   {"key": "dev_75bc0c09", "task": "SLU_scenario", "target": "news", "path": "/data/slurp/wavs/dev_75bc0c09.wav"}
   ```
   Tasks supported: ASR, EN2ZH, EN2DE, QA, SLU_scenario (SLURP).  
   (For more tasks, add corresponding prompts in `/conf/multiprompt.jsonl`.)

3. One-Click Script:
Core training script: `/scripts/finetune_deepspeed_sensevoice.sh`
Inference script: `/scripts/decode_sensevoice.sh`
## Core Parameter Explanation
| Variable | Value | Purpose |
|---|---|---|
| `TOKENIZERS_PARALLELISM=false` | Disable HuggingFace tokenizer parallelism | Avoid deadlock |
| `HCCL_CONNECT_TIMEOUT=7200` | Ascend NCCL timeout 2 h | Large-model comm tolerance |
| `ASCEND_LAUNCH_BLOCKING=1` | Ascend sync execution | Easier OOM / operator debug |
| `CPU_AFFINITY_CONF=2` | Fine-grained core binding | Reduce context switch |
| `OMP_NUM_THREADS=1` | Limit OpenMP threads | Prevent CPU oversubscription |
| `multitask_prompt_path` | `conf/multiprompt.jsonl` | Prompt templates per task |
| `llm_path` | `/.../Qwen2.5-1.5B-Instruct` | LLM weight directory |
| `llm_name` | `Qwen2.5-1.5B-Instruct` | Large-language-model choice |
| `projector` | `linear-silu` | Projector type |
| `encoder_dim=25055` | 25055 (SenseVoiceSmall) | Encoder output dimension |
| `speech_encoder_path` | `/.../SenseVoiceSmall` | Encoder weight directory |
| `encoder_name` | `sensevoice` | Speech encoder choice |
| `use_peft` | `false` | Whether to use LoRA fine-tuning |
| `gt_emb` | `true` | Use text embedding |
| `gt_emb_noise` | `true` | Smooth GT embedding |
| `freeze_encoder` | `true` | Freeze encoder |
| `freeze_projector` | `false` | Freeze projector |
| `do_psd` | `true` | Enable PSD |
| `ctc_posterior` | `true` | Use CTC posterior |
| `voca_trans` | `false` | true = LegoSLM baseline (ctc posterior * llm_emb_matrix) |
| `use_dynamic_sampling` | `false` | Dynamic sampling (not supported yet) |
| `validation_interval` | `1000` | Validation interval |
| `num_epochs` | `5` | Number of training epochs |
| `train_scp_file_path` | `...` | Training file path (directory must contain `multitask.jsonl`) |
| `dev_scp_file_path` | `...` | Validation file path (directory must contain `multitask.jsonl`) |


---

## 📖 Citation

If you find **TASU** or this codebase useful in your research, please consider citing:

```bibtex
@article{peng2025tasu,
  title   = {TASU: Text-Only Alignment for Speech Understanding},
  author  = {Peng, Jing and Yang, Yi and Li, Xu and Xi, Yu and Tang, Quanwei and Fang, Yangui and Li, Junjie and Yu, Kai},
  journal = {arXiv preprint arXiv:2511.03310},
  year    = {2025},
}
``` 
