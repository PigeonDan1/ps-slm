import argparse
import os
import torch
import torch_npu
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperFeatureExtractor
from model import WhisperCTC
from dataset import JsonlCTCDataset, ctc_collate_fn, Vocabulary
from search import ctc_greedy_search, ctc_prefix_beam_search

def decode_results(results, tokenizer, use_vocab=False):
    texts = []
    for r in results:
        if use_vocab:
            text = tokenizer.sequence_to_text(r)
        else:
            text = tokenizer.decode(r, skip_special_tokens=True)
        texts.append(text)
    return texts

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to jsonl data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--mode", type=str, default="greedy")
    parser.add_argument("--tokenizer_path", type=str, default="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--vocab_file", type=str, default=None)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else f"npu:{args.device}"
    mode = args.mode
    print("Loading Tokenizer and FeatureExtractor...")
    if args.vocab_file:
        tokenizer = Vocabulary(args.vocab_file)
        use_vocab = True
        vocab_size = len(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        use_vocab = False
        vocab_size = tokenizer.vocab_size

    feature_extractor = WhisperFeatureExtractor.from_pretrained("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/openai-mirror/whisper-medium")
    print("Loading Model...")
    model = WhisperCTC("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/openai-mirror/whisper-medium", vocab_size=vocab_size+1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print("Preparing DataLoader...")
    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor, mode = "test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ctc_collate_fn)

    ckpt_dir = os.path.dirname(args.checkpoint)
    pred_output_path = os.path.join(ckpt_dir, f"pred_{mode}.txt")
    gt_output_path = os.path.join(ckpt_dir, f"gt_{mode}.txt")

    print(f"Pred output will be saved to: {pred_output_path}")
    print(f"GT output will be saved to: {gt_output_path}")

    print("Starting Inference...")
    blank_id = vocab_size  # same as CTC blank setting

    with open(pred_output_path, 'w', encoding='utf-8') as pred_fout, \
         open(gt_output_path, 'w', encoding='utf-8') as gt_fout:
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                uids = batch["uid"]
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                input_lens = batch["input_lengths"].to(device)
                logits = model(input_features, attention_mask=attention_mask)
                log_probs = logits.log_softmax(dim=-1)
                if mode == "greedy":
                    cleaned_ids = ctc_greedy_search(log_probs, input_lens, blank_id=blank_id)
                else:
                    cleaned_ids = ctc_prefix_beam_search(log_probs, input_lens, blank_id=blank_id, beam_size=10)
                decoded_text = decode_results(cleaned_ids, tokenizer, use_vocab=use_vocab)

                for i in range(len(decoded_text)):
                    gt_labels = labels[i]
                    gt_labels = gt_labels[gt_labels >= 0]
                    if use_vocab:
                        gt_text = tokenizer.sequence_to_text(gt_labels.tolist())
                    else:
                        gt_text = tokenizer.decode(gt_labels, skip_special_tokens=True)
                    uid = uids[i]
                    print(f"[Sample {batch_idx}] uid={uid}")
                    print(f"  pred : {decoded_text[i]}")
                    print(f"  gt   : {gt_text}")
                    pred_fout.write(f"{uid} {decoded_text[i]}\n")
                    gt_fout.write(f"{uid} {gt_text}\n")

if __name__ == "__main__":
    main()
