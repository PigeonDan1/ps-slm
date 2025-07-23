# LegoSLM Reproduction

This repository contains code for reproducing the LegoSLM project. It is organized into two main components:

## 📌 1. SLAM-LLM-ASR-Whisper
A Speech Large Language Model (Speech LLM) built on the SLAM-LLM-ASR framework, using the Whisper model as the encoder. This component integrates the Whisper encoder into a modular speech-to-text system designed for instruction-following and conversational ASR.

## 📌 2. Whisper-CTC
A standalone module for training the Whisper encoder using a Connectionist Temporal Classification (CTC) loss. This part focuses on fine-tuning Whisper’s encoder representations. Now we support sentence_piece/ gemma-2b / qwen-2.5 / self vocabulary.
1. Deepspeed brand is not ready yet, don't use train_ddp.etc.
2. Environment of SLAM-LLM-NPU is fully suppported.

