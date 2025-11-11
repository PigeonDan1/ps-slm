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

> Full GPU-specific support is under active development (actually is almost ready), and we plan to provide more detailed GPU setup and scripts soon.

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

3. Download pre-trained models and our checkpoints(huggingFace):

   SenseVoiceSmall:https://huggingface.co/FunAudioLLM/SenseVoiceSmall

   Qwen2.5-1.5B:https://huggingface.co/Qwen/Qwen2.5-1.5B

   ckpts:https://huggingface.co/yyy1421129/ps-slm https://www.modelscope.cn/models/yyy1421129/ps-slm
    - text_only: Checkpoint trained with only text
    - half_audio_finetuned: SFT using 900h audio based on text_only/pytorch_model.bin
    - Method to use these ckpts: Download and Fill in the ckpt_path variable in scripts scripts/decode_sensevoice.sh with the path to the downloaded model checkpoint.
   
5. One-Click Script:
   
   Core training script: `/scripts/finetune_deepspeed_sensevoice.sh`

   Inference script: `/scripts/decode_sensevoice.sh`

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
