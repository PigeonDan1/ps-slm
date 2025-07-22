import os, argparse, torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, WhisperFeatureExtractor
from tqdm import tqdm
import json
import deepspeed
from dataset import JsonlCTCDataset, ctc_collate_fn
from model import WhisperCTC


@torch.no_grad()
def evaluate(model, dataloader, device, ctc_loss_fn, rank):
    model.eval()
    total_loss = 0
    total_batches = 0
    for batch in dataloader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)

        logits = model(input_features, attention_mask=attention_mask)
        log_probs = logits.log_softmax(2).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

        total_loss += loss.item()
        total_batches += 1
    model.train()
    avg_loss = total_loss / max(1, total_batches)
    if rank == 0:
        print(f"[Eval] Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train(args):
    # 设置分布式参数
    use_npu = not torch.cuda.is_available() and hasattr(torch, "npu")

    # 初始化分布式后端
    if use_npu:
        import torch_npu
        dist_backend = "hccl"
    else:
        dist_backend = "nccl"

    dist.init_process_group(backend=dist_backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置 device
    if use_npu:
        import torch_npu
        device = torch.device(f"npu:{args.local_rank}")
        torch.npu.set_device(device)
    else:
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    blank_id = tokenizer.vocab_size
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)

    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=ctc_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    valid_dataloader = None
    if args.valid_jsonl:
        valid_dataset = JsonlCTCDataset(args.valid_jsonl, tokenizer, feature_extractor)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler,
            collate_fn=ctc_collate_fn,
            num_workers=4,
            pin_memory=True
        )

    # 构造模型
    model = WhisperCTC(
        whisper_name=args.whisper_path,
        vocab_size=tokenizer.vocab_size + 1,
        freeze_encoder=False
    )

    # DeepSpeed 接管 optimizer 和 scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    with open(args.deepspeed, "r") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * ds_config["gradient_accumulation_steps"] * dist.get_world_size()
    model, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        config_params=ds_config
    )

    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
        else:
            pbar = dataloader

        for batch in pbar:
            input_features = batch["input_features"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)

            logits = model(input_features, attention_mask=attention_mask)
            log_probs = logits.log_softmax(2).transpose(0, 1)
            loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

            model.backward(loss)
            model.step()

            global_step += 1
            if rank == 0:
                pbar.set_postfix(loss=loss.item())

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if valid_dataloader:
                        evaluate(model, valid_dataloader, device, ctc_loss_fn, rank)
                    ckpt_path = args.save_path.replace(".pt", f"_step{global_step}.pt")
                    model.save_checkpoint(os.path.dirname(ckpt_path), tag=f"step{global_step}")

        if valid_dataloader:
            evaluate(model, valid_dataloader, device, ctc_loss_fn, rank)
        if rank == 0:
            ckpt_path = args.save_path.replace(".pt", f"_epoch{epoch}.pt")
            model.save_checkpoint(os.path.dirname(ckpt_path), tag=f"epoch{epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--valid_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_path", default="whisper_ctc.pt")
    parser.add_argument("--tokenizer_path", default="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--whisper_path", default="/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/openai-mirror/whisper-medium")
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--deepspeed", default="ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1, help="For deepspeed compatibility")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
