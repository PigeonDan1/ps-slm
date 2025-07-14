import argparse
import torch
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

    device = "cuda" if torch.cuda.is_available() else f"npu:{device}"

    print("Loading Tokenizer and FeatureExtractor...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    print("Loading Model...")
    model = WhisperCTC("openai/whisper-large-v3", vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    print("Preparing DataLoader...")
    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ctc_collate_fn)

    print("Starting Inference...")
    blank_id = 0  # same as CTC blank setting

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_features = batch["input_features"].to(device)
            logits = model(input_features)
            log_probs = logits.softmax(dim=-1)
            pred_ids = log_probs.argmax(dim=-1)  # (B, T)

            for i in range(pred_ids.size(0)):
                raw_ids = pred_ids[i]
                cleaned_ids = ctc_greedy_decode(raw_ids, blank_id=blank_id)
                decoded_text = tokenizer.decode(cleaned_ids, skip_special_tokens=True)
                print(f"[Sample {batch_idx}] {decoded_text}")

if __name__ == "__main__":
    main()
