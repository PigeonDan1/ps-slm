import os, argparse, torch, torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, WhisperFeatureExtractor
from tqdm import tqdm
from dataset import JsonlCTCDataset, ctc_collate_fn, Vocabulary, KaldiFbankExtractor
from model import WhisperCTC, SenseVoiceCTC

def setup(rank, world_size, backend="hccl"):
    """初始化通信后端：NPU 用 hccl，GPU 用 nccl"""
    if torch.cuda.is_available():
        backend = "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

@torch.no_grad()
def evaluate(model, dataloader, device, ctc_loss_fn, rank, encoder_name):
    model.eval()
    total_loss = 0
    total_batches = 0
    for batch in dataloader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        label_lengths = batch["label_lengths"].to(device, non_blocking=True)
        if encoder_name == "whisper":
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            # print(attention_mask) 
            logits = model(input_features, attention_mask)
            log_probs = logits.log_softmax(2).transpose(0, 1) # (1500, 4, 151644)
        else:
            logits, input_lengths = model(input_features, input_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]
        loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    avg_loss = total_loss / max(1, total_batches)
    if rank == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train(rank, world_size, args):
    setup(rank, world_size)
    print("Here is the arguments: ")
    print(args)
    print("Check possible probelms !!!")
    # 决定当前进程绑定的设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"npu:{rank}")
        torch_npu.npu.set_device(device)

    # Tokenizer / Feature Extractor 每个进程都加载（可优化为 rank0 加载后 broadcast）
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # blank_id = tokenizer.vocab_size
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
            args.whisper_path,
            vocab_size=vocab_size,
            freeze_encoder=True
        )
    elif args.encoder_name == "sensevoice":
        from model import SenseVoiceSmall
        model, kwargs = SenseVoiceSmall.from_pretrained(args.encoder_path)
        model = SenseVoiceCTC(model, vocab_size=vocab_size, freeze_encoder=True)
        feature_extractor = KaldiFbankExtractor(kwargs)

    dataset = JsonlCTCDataset(args.jsonl, tokenizer, feature_extractor, mode = "train", encoder=args.encoder_name)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=ctc_collate_fn,
        num_workers=0,
        pin_memory=False
    )
    if args.valid_jsonl:
        valid_dataset = JsonlCTCDataset(args.valid_jsonl, tokenizer, feature_extractor, mode="dev", encoder=args.encoder_name)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler,
            collate_fn=ctc_collate_fn,
            num_workers=0,
            pin_memory=False
        )
    else:
        valid_dataloader = None

    model.to(device)
    model = DDP(model, device_ids=[rank])
    import os
    os.makedirs(args.save_path, exist_ok=True)
    # 优化器（weight_decay 可按需再调）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01
    )

    # 总步数估算：len(dataloader) * epochs
    total_steps = len(dataloader) * args.epochs
    warmup_steps = min(2000, total_steps // 10)          # 10 % 或最多 1k

    # 构造 Lambda 调度器：先线性 warmup，再余弦退火
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # 余弦退火到 0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        for batch in pbar:
            optimizer.zero_grad()

            input_features = batch["input_features"].to(device, non_blocking=True) # (4, 80, 3000)
            labels = batch["labels"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True) # (len1, ..., len4)
            label_lengths = batch["label_lengths"].to(device, non_blocking=True)
            if args.encoder_name == "whisper":
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                # print(attention_mask) 
                logits = model(input_features, attention_mask)
                log_probs = logits.log_softmax(2).transpose(0, 1) # (1500, 4, 151644)
            else:
                # print(f"input_lens: {input_lengths}")
                logits, input_lengths = model(input_features, input_lengths)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, V]

            loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            if rank == 0:
                pbar.set_postfix(loss=loss.item())
                # 每 N steps 验证+保存
                if global_step % args.eval_steps == 0:
                    if valid_dataloader is not None:
                        val_loss = evaluate(model, valid_dataloader, device, ctc_loss_fn, rank, args.encoder_name)
                    ckpt_name = os.path.join(args.save_path, f'step_{global_step}.pt')
                    torch.save(model.module.state_dict(), ckpt_name)

        # 评估+保存
        if valid_dataloader is not None:
            evaluate(model, valid_dataloader, device, ctc_loss_fn, rank, args.encoder_name)
        if rank == 0:
            save_name = os.path.join(args.save_path, f'epoch_{epoch}.pt')
            torch.save(model.module.state_dict(), save_name)

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_path", default="whisper_ctc.pt")
    parser.add_argument("--tokenizer_path", default="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--encoder_name", default="whisper")
    parser.add_argument("--encoder_path", default="/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/openai-mirror/whisper-medium")
    parser.add_argument("--valid_jsonl", required=True)
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run eval/save every N steps")
    parser.add_argument("--vocab_file", default=None, help="Path to vocab.txt. If set, will override tokenizer.")

    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    train(rank, world_size, args)

if __name__ == "__main__":
    main()
