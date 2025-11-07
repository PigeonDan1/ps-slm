<details>
<summary>📖 English Version</summary>

# 📌 Data Format:
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

{"key": "common_voice_en_211671", "task": "ASR", "target": "That is a weird phrase.", "path": "/data/audio/dev/common_voice_en_211671.wav"}

{"key": "dev_75bc0c09", "task": "SLU_scenario", "target": "news", "path": "/data/slurp/wavs/dev_75bc0c09.wav"}

Tasks supported: ASR, EN2ZH, EN2DE, QA, SLU_scenario (SLURP).  
(For more tasks, add corresponding prompts in `/conf/multiprompt.jsonl`.)

# 📌 One-Click Script:
Core training script: `/scripts/finetune_deepspeed_sensevoice.sh`

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

Inference script: `/scripts/decode_sensevoice.sh`

<details>
<summary>📖 English Version</summary>

## 📌 数据格式：
每条样本是 **一行合法 JSON（JSON Lines）**，字段名与取值约束如下：
| 字段  | 类型   | 必填 | 说明 |
|-------|--------|------|------|
| key   | string | ✔    | 全局唯一 ID，禁止含 `/` 或空格 |
| task  | string | ✔    | 任务代号（ASR，EN2ZH等） |
| target| string | ✔    | 模型需输出的文本（label / decoding target） |
| path  | string | ✔    | 音频位置，支持 2 种协议，见下文 |
| GT    | string | ✘(✔) | 音频GT用于文本仿真CTC后验 |

音频格式支持：
| 协议     | 示例路径                                | 读取提示                |
| ------ | ----------------------------------- | ------------------- |
| 普通 wav | `/xxx/common_voice_en_19641841.wav` | 直接 `soundfile.read` |
| ark 偏移 | `/xxx/data_wav.1.ark:246511401`     | 二进制 `seek(offset)`  |

数据示例：

{"key": "common_voice_en_211671", "task": "ASR", "target": "That is a weird phrase.", "path": "/data/audio/dev/common_voice_en_211671.wav"}

{"key": "dev_75bc0c09", "task": "SLU_scenario", "target": "news", "path": "/data/slurp/wavs/dev_75bc0c09.wav"}

任务支持：ASR, EN2ZH, EN2DE, QA, SLU_scenario(SLURP) (如果需要支持更多任务，请在/conf/multiprompt.jsonl里加入相应的prompt)

## 📌 脚本一键运行：
核心训练脚本：/scripts/finetune_deepspeed_sensevoice.sh
## 核心参数解释
| 变量 | 取值 | 作用 |
|---|---|---|
| `TOKENIZERS_PARALLELISM=false` | 关闭 HuggingFace tokenizer 并行 | 避免死锁 |
| `HCCL_CONNECT_TIMEOUT=7200` | 昇腾 NCCL 超时 2 h | 大模型通信容错 |
| `ASCEND_LAUNCH_BLOCKING=1` | 昇腾同步执行 | 方便定位 OOM / 算子错误 |
| `CPU_AFFINITY_CONF=2` | 细粒度绑核 | 减少上下文切换 |
| `OMP_NUM_THREADS=1` | 限制 OpenMP 线程 | 防止 CPU 抢占 |
| `multitask_prompt_path` | `conf/multiprompt.jsonl` | 不同任务对应的 prompt 模板 |
| `llm_path` | `/.../Qwen2.5-1.5B-Instruct` | LLM 权重目录 |
| `llm_name` | `Qwen2.5-1.5B-Instruct` | 大语言模型选型 |
| `projector` | `linear-silu` | 投影层类型 |
| `encoder_dim=25055` | 25055(senseVoiceSmall) | 编码器输出维度 |
| `speech_encoder_path` | `/.../SenseVoiceSmall` | 编码器权重目录 |
| `encoder_name` | `sensevoice` | 语音编码器选型 |
| `use_peft` | `false`  | 是否使用lora微调 |
| `gt_emb` | `true` | 使用文本embedding |
| `gt_emb_noise` | `true` | 对GT embedding平滑 |
| `freeze_encoder` | `true` | 是否冻结encoder |
| `freeze_projector` | `false` | 是否冻结projector |
| `do_psd` | `true` | 是否启动PSD |
| `ctc_posterior` | `true` | 是否使用ctc后验 |
| `voca_trans` | `false` | true为LegoSLM基线（ctc后验*llm_emb_matrix） |
| `use_dynamic_sampling` | `false` | 动态采样（暂未支持） |
| `validation_interval` | `1000` | 验证间隔 |
| `num_epochs` | `5` | 训练轮数 |
| `train_scp_file_path` | `...` | 训练文件路径（路径下需要有multitask.jsonl文件）|
| `dev_scp_file_path` | `...` | 验证文件路径（路径下需要有multitask.jsonl文件）|

推理脚本：/scripts/decode_sensevoice.sh
