import argparse
import os
import torch
import torch_npu
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperFeatureExtractor
from model import WhisperCTC
from dataset import JsonlCTCDataset, ctc_collate_fn
from utils import ctc_greedy_decode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to jsonl data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='0')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else f"npu:{args.device}"

    print("Loading Tokenizer and FeatureExtractor...")
    tokenizer = AutoTokenizer.from_pretrained("/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/whisper-large-v3")

    print("Loading Model...")
    model = WhisperCTC("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/whisper-large-v3", vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    print("Preparing DataLoader...")
    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ctc_collate_fn)

    # 确定输出路径
    ckpt_dir = os.path.dirname(args.checkpoint)
    pred_output_path = os.path.join(ckpt_dir, "pred.txt")
    gt_output_path = os.path.join(ckpt_dir, "gt.txt")

    print(f"Pred output will be saved to: {pred_output_path}")
    print(f"GT output will be saved to: {gt_output_path}")

    print("Starting Inference...")
    blank_id = tokenizer.vocab_size  # same as CTC blank setting

    with open(pred_output_path, 'w', encoding='utf-8') as pred_fout, \
         open(gt_output_path, 'w', encoding='utf-8') as gt_fout:
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_features)
                log_probs = logits.softmax(dim=-1)
                pred_ids = log_probs.argmax(dim=-1)  # (B, T)

                for i in range(pred_ids.size(0)):
                    raw_ids = pred_ids[i]
                    cleaned_ids = ctc_greedy_decode(raw_ids, blank_id=blank_id)
                    decoded_text = tokenizer.decode(cleaned_ids, skip_special_tokens=True)
                    gt_labels = labels[i]
                    gt_labels = gt_labels[gt_labels >= 0]
                    gt_text = tokenizer.decode(gt_labels, skip_special_tokens=True)

                    # 打印到屏幕
                    print(f"[Sample {batch_idx}]")
                    print(f"  pred : {decoded_text}")
                    print(f"  gt   : {gt_text}")

                    # 分别写到两个文件
                    pred_fout.write(f"{decoded_text}\n")
                    gt_fout.write(f"{gt_text}\n")

if __name__ == "__main__":
    main()
