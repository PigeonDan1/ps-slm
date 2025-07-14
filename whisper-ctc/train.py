import argparse
import torch
import torch_npu
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperFeatureExtractor
from tqdm import tqdm

from dataset import JsonlCTCDataset, ctc_collate_fn
from model import WhisperCTC

def train_ctc(model, dataloader, optimizer, device, epochs=5):
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            input_lengths = batch["input_lengths"]
            label_lengths = batch["label_lengths"]

            logits = model(input_features)
            log_probs = logits.log_softmax(2).transpose(0, 1)

            loss = ctc_loss_fn(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        if device.startswith("npu"):
            torch_npu.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, help="Path to training jsonl", default="/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/export_llm/multitask.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument("--save_path", type=str, default="whisper_ctc.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else f"npu:{args.device}"

    print("Loading Tokenizer and Feature Extractor...")
    tokenizer = AutoTokenizer.from_pretrained("/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/whisper-large-v3")

    print("Preparing Dataset...")
    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn)

    print("Building Model...")
    model = WhisperCTC("/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/whisper-large-v3", vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting Training...")
    train_ctc(model, dataloader, optimizer, device, epochs=args.epochs)

    print(f"Saving Model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Done.")
