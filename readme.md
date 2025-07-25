# PS-SLM

<details>
<summary>📖 English Version</summary>

## Overview

**PS-SLM** (*Phone-Synchronized Speech Language Model*) introduces a novel alignment strategy based on **phone-synchronized decoding**, extending prior works like **LegoSLM** and **SLAM-LLM**. It improves speech-text alignment and enables more effective integration between encoders and language models.

This repository consists of two main components:

---

## 📌 1. SLAM-LLM-ASR-Whisper

An instruction-following **Speech Large Language Model (Speech LLM)** based on the SLAM-LLM-ASR framework. It incorporates the **Whisper encoder** into a modular ASR pipeline enhanced with **phone-synchronized alignment**, improving transcription performance and contextual understanding.

---

## 📌 2. Whisper-CTC

A standalone CTC training module for fine-tuning the Whisper encoder. It supports both standard and phone-synchronized alignment strategies.

### Key Features:
- Encoder + CTC training pipeline.
- Supports vocabularies including `sentence_piece`, `gemma-2b`, `qwen-2.5`, and self-defined token sets.
- Flexible freezing/unfreezing of the Whisper encoder.
- Compatible with downstream SLAM-style tasks.

> ⚠️ **Notes:**
> - Deepspeed-based distributed training is **not yet supported** — avoid using `train_deepspeed.*` scripts.  
> - The **SLAM-LLM-NPU** environment is **fully supported**.

</details>

---

<details>
<summary>📘 中文版本</summary>

## 项目简介

**PS-SLM**（Phone-Synchronized Speech Language Model）提出了一种基于**音素同步解码（Phone-Synchronized Decoding）**的新型对齐方式，进一步改进了 **LegoSLM** 与 **SLAM-LLM** 框架中原有的对齐策略，提升了语音与文本的对齐质量，并加强了编码器与语言模型之间的融合能力。

本项目主要包含两个核心模块：

---

## 📌 1. SLAM-LLM-ASR-Whisper

一个基于 SLAM-LLM-ASR 框架的指令遵循型 **语音大语言模型（Speech LLM）**。该模块集成了 **Whisper 编码器**，采用模块化设计，支持基于 **音素同步对齐** 的增强型转录与语境理解能力。

---

## 📌 2. Whisper-CTC

独立的 CTC 训练模块，用于微调 Whisper 编码器，支持传统对齐与**音素同步对齐**两种方式。

### 主要特性：
- 支持 Encoder + CTC 的训练范式；
- 兼容多种词表：`sentence_piece`、`gemma-2b`、`qwen-2.5`、自定义词表等；
- 支持 Whisper 编码器参数的灵活冻结与解冻；
- 可与 SLAM 风格任务无缝对接。

> ⚠️ **注意事项：**  
> - 当前暂不支持基于 Deepspeed 的分布式训练，**请勿使用 `train_deepspeed.*` 系列脚本**；  
> - 已全面适配并验证 **SLAM-LLM-NPU 环境**。

</details>
