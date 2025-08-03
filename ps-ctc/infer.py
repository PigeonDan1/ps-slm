import argparse
import os
import torch
import torch_npu
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperFeatureExtractor
from dataset import JsonlCTCDataset, ctc_collate_fn, Vocabulary, KaldiFbankExtractor
from model import WhisperCTC, SenseVoiceCTC
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
    parser.add_argument("--split", type=str, required=True, help="The test split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--mode", type=str, default="greedy")
    parser.add_argument("--tokenizer_path", type=str, default="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--encoder_name", default="whisper")
    parser.add_argument("--encoder_path", default="/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/openai-mirror/whisper-medium")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else f"npu:{args.device}"
    print(f"Check your args: {args}")
    mode = args.mode
    split = args.split
    print("Loading Tokenizer and FeatureExtractor...")
    if args.vocab_file:
        tokenizer = Vocabulary(args.vocab_file)
        blank_id = len(tokenizer)  # 假设 blank 放在 vocab_size 的位置
        use_custom_vocab = True
        print(f"Load vocabulary file from:{args.vocab_file}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        blank_id = tokenizer.vocab_size
        use_custom_vocab = False
        print(f"Load Tokenizer from: {args.tokenizer_path}")
    vocab_size = len(tokenizer) + 1 if use_custom_vocab else tokenizer.vocab_size + 1
    if args.encoder_name == "whisper":
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.encoder_path)
        model = WhisperCTC(
            args.encoder_path,
            vocab_size=vocab_size,
            freeze_encoder=True
        )
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    elif args.encoder_name == "sensevoice":
        from model import SenseVoiceSmall
        model, kwargs = SenseVoiceSmall.from_pretrained(args.encoder_path)
        model = SenseVoiceCTC(model, vocab_size=vocab_size, freeze_encoder=True)
        feature_extractor = KaldiFbankExtractor(kwargs)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    print("Preparing DataLoader...")
    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor, mode = "train", encoder=args.encoder_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ctc_collate_fn)

    ckpt_dir = os.path.dirname(args.checkpoint)
    pred_output_path = os.path.join(ckpt_dir, f"{split}_pred_{mode}.txt")
    gt_output_path = os.path.join(ckpt_dir, f"{split}_gt_{mode}.txt")

    print(f"Pred output will be saved to: {pred_output_path}")
    print(f"GT output will be saved to: {gt_output_path}")

    print("Starting Inference...")

    with open(pred_output_path, 'w', encoding='utf-8') as pred_fout, \
         open(gt_output_path, 'w', encoding='utf-8') as gt_fout:
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                uids = batch["uid"]
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                input_lens = batch["input_lengths"].to(device)
                if args.encoder_name == "whisper":
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    # print(attention_mask) 
                    logits = model(input_features, attention_mask)
                    log_probs = logits.log_softmax(2) 
                else:
                    # print(f"input_lens: {input_lengths}")
                    logits, input_lens = model(input_features, input_lens)
                    log_probs = logits.log_softmax(dim=-1)  # [B, T, V]

                if mode == "greedy":
                    cleaned_ids = ctc_greedy_search(log_probs, input_lens, blank_id=blank_id)
                else:
                    cleaned_ids = ctc_prefix_beam_search(log_probs, input_lens, blank_id=blank_id, beam_size=10)
                decoded_text = decode_results(cleaned_ids, tokenizer, use_vocab=use_custom_vocab)

                for i in range(len(decoded_text)):
                    gt_labels = labels[i]
                    gt_labels = gt_labels[gt_labels >= 0]
                    if use_custom_vocab:
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
