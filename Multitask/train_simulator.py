import sys
import os
import logging
import random
import torch
import torch.distributed as dist
import deepspeed
import torch_npu
import editdistance
from torch_npu.contrib import transfer_to_npu
from tqdm import tqdm

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra

from simulator.factory import create_training_model
from simulator.dataset import SimulatorDataset, SimulatorCollate
from utils.deepspeed_utils import clear_gpu_cache, setup_environ_flags

# --- 参数清洗逻辑 ---
cleaned_argv = []
i = 0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg.startswith("--local_rank"):
        if "=" in arg:
            if len(arg.split("=")) > 1:
                val = arg.split("=")[1]
                os.environ["LOCAL_RANK"] = val
            i += 1 
        else:
            if i + 1 < len(sys.argv):
                val = sys.argv[i+1]
                os.environ["LOCAL_RANK"] = val
                i += 2 
            else:
                i += 1
    else:
        cleaned_argv.append(arg)
        i += 1
sys.argv = cleaned_argv

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False

logger = logging.getLogger(__name__)

# 用于反馈计算WER
class Calculator:
    def __init__(self):
        self.space = []
        self.cost = {'cor': 0, 'sub': 1, 'del': 1, 'ins': 1}

    def calculate(self, lab_raw, rec_raw):
        # 严格对齐 pt.py 逻辑：转为字符串比对
        lab = [str(x) for x in lab_raw]
        rec = [str(x) for x in rec_raw]
        lab.insert(0, ''); rec.insert(0, '')
        
        # 动态扩展矩阵空间
        while len(self.space) < len(lab): self.space.append([])
        for row in self.space:
            while len(row) < len(rec): row.append({'dist': 0, 'error': 'non'})
            
        # 初始化边界
        for i in range(len(lab)): self.space[i][0]['dist'] = i; self.space[i][0]['error'] = 'del'
        for j in range(len(rec)): self.space[0][j]['dist'] = j; self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        
        for i in range(1, len(lab)):
            for j in range(1, len(rec)):
                min_dist = sys.maxsize
                d = self.space[i-1][j]['dist'] + self.cost['del']
                if d < min_dist: min_dist, err = d, 'del'
                d = self.space[i][j-1]['dist'] + self.cost['ins']
                if d < min_dist: min_dist, err = d, 'ins'
                c = self.cost['cor'] if lab[i] == rec[j] else self.cost['sub']
                d = self.space[i-1][j-1]['dist'] + c
                if d < min_dist: min_dist, err = d, ('cor' if lab[i] == rec[j] else 'sub')
                self.space[i][j]['dist'] = min_dist; self.space[i][j]['error'] = err
        
        # 回溯 SDI 指标
        res = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        i, j = len(lab)-1, len(rec)-1
        while True:
            op = self.space[i][j]['error']
            if op == 'cor': res['all'] += 1; res['cor'] += 1; i -= 1; j -= 1
            elif op == 'sub': res['all'] += 1; res['sub'] += 1; i -= 1; j -= 1
            elif op == 'del': res['all'] += 1; res['del'] += 1; i -= 1
            elif op == 'ins': res['ins'] += 1; j -= 1
            else: break
        return res

def compute_batch_penalty(sampled_ids_list, batch_text_ids, error_stats, calculator):
    """
    sampled_ids_list: List[Tensor], 长度为 K, 每个 Tensor 维度 [B, T_k]
    返回: [B, K] 的张量，记录每条路径的 SDI 偏差 (常数)
    """
    B = batch_text_ids.size(0)
    K = len(sampled_ids_list)
    penalties = torch.zeros(B, K)
    
    for k in range(K):
        gen_ids = sampled_ids_list[k]
        for i in range(B):
            hyp = [x.item() for x in gen_ids[i] if x != 0 and x != 25055]
            ref = [x.item() for x in batch_text_ids[i] if x != 0 and x != 25055]
            
            res = calculator.calculate(ref, hyp)
            n = res['all']
            if n == 0:
                penalties[i, k] = 0.0
                continue
                
            actual_stats = [
                res['sub'] / n, res['del'] / n, res['ins'] / n, 
                (res['sub'] + res['del'] + res['ins']) / n
            ]
            target_stats = error_stats[i].cpu().numpy()
            
            # 计算 SDI + WER 的总绝对偏差
            dev = sum(abs(a - t) for a, t in zip(actual_stats, target_stats))
            penalties[i, k] = dev
            
    return penalties

@dataclass
class RunConfig:
    model_config: dict = field(default_factory=dict)
    dataset_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    log_config: dict = field(default_factory=dict)
    deepspeed_config: str = ""
    # [新增] 预训练权重路径
    ckpt_path: str = "" 

@hydra.main(config_path="conf", config_name="simulator_config", version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    main(cfg)

def main(kwargs: DictConfig):
    train_config = kwargs.train_config
    model_config = kwargs.model_config
    dataset_config = kwargs.dataset_config
    log_config = kwargs.log_config
    ds_config_path = kwargs.deepspeed_config
    ckpt_path = kwargs.ckpt_path # [新增]

    # --- 环境设置 ---
    setup_logging(log_config, train_config.output_dir)
    
    step_logger = None
    rank = int(os.environ.get("RANK", -1))
    if rank == 0:
        step_logger = logging.getLogger("step_logger")
        step_logger.propagate = False 
        step_logger.setLevel(logging.INFO)
        log_path = os.path.join(train_config.output_dir, "train.log")
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        step_logger.addHandler(fh)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.npu.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend="hccl")

    set_seed(train_config.seed)

    if rank == 0:
        logger.info(f"Training Config: {OmegaConf.to_yaml(train_config)}")
        
    # --- 数据集构建 ---
    logger.info(f"Loading datasets... (Rank {rank})")
    train_dataset = SimulatorDataset(dataset_config.train_path, max_len=model_config.max_len)
    
    dev_dataset = None
    if train_config.get("run_validation", False):
        dev_dataset = SimulatorDataset(dataset_config.dev_path, max_len=model_config.max_len)
    
    collate_fn = SimulatorCollate(max_len=model_config.max_len)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size_per_gpu,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True
    )

    dev_dataloader = None
    if dev_dataset:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset, shuffle=False, drop_last=False
        )
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=train_config.batch_size_per_gpu, 
            sampler=dev_sampler,
            collate_fn=collate_fn,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True
        )

    # --- 模型构建 ---
    logger.info(f"Creating model... (Rank {rank})")
    model_wrapper, tokenizer = create_training_model(model_config, dataset_config)
    
    # ================= [新增] 加载预训练权重 =================
    if ckpt_path and os.path.exists(ckpt_path):
        if rank == 0:
            logger.info(f"--> Loading pretrained checkpoint from: {ckpt_path}")
        
        # 加载权重到 CPU
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # 处理可能的键名不匹配 (如果旧ckpt里有 module. 前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            elif k.startswith("simulator."): 
                # 旧 wrapper 可能直接存了 simulator.xxx，如果你现在结构没变，直接用
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v
        
        # [核心] strict=False
        # 允许加载 Encoder/Decoder 权重，同时忽略新加的 condition_projector (随机初始化)
        missing_keys, unexpected_keys = model_wrapper.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0:
            logger.info(f"--> Missing keys (Initialized randomly): {missing_keys}")
            logger.info(f"--> Unexpected keys (Ignored): {unexpected_keys}")
    else:
        if rank == 0 and ckpt_path:
            logger.warning(f"--> Checkpoint path provided but not found: {ckpt_path}")
    # ========================================================

    if rank == 0:
        total_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
        logger.info(f"--> Model Parameters: {total_params / 1e6:.2f} Million")

    # --- DeepSpeed 初始化 ---
    logger.info(f"Initializing DeepSpeed... (Rank {rank})")
    parameters = filter(lambda p: p.requires_grad, model_wrapper.parameters())
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model_wrapper,
        model_parameters=parameters,
        config=ds_config_path
    )

    # --- 训练循环 ---
    global_step = 0
    best_loss = float('inf')
    num_epochs = train_config.num_epochs

    calc = Calculator()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        
        pbar = None
        if rank == 0:
            logger.info(f"=== Starting Epoch {epoch+1}/{num_epochs} ===")
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        # 训练循环内部逻辑片段
        # 训练循环内部逻辑片段 (适配优化后的并行 MBR)
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, model_engine.device)
            
            # --- 步骤 1: 基础 SFT (Teacher Forcing) ---
            outputs, acc = model_engine(
                text_onehot=batch["text_onehot"],
                text_mask=batch["text_mask"],
                target_psd=batch["target_psd"],
                target_lengths=batch["target_lengths"],
                error_stats=batch["error_stats"]
            )
            loss_ce = outputs.loss

            use_mbr = getattr(model_config, "use_mbr", True)
            mbr_lambda = getattr(model_config, "mbr_lambda", 2.0)
            mbr_tau = getattr(model_config, "mbr_tau", 0.8)
            mbr_k = getattr(model_config, "mbr_k", 3)

            if use_mbr:
                raw_model = model_engine.module.simulator if hasattr(model_engine.module, "simulator") else model_engine.module
                
                # --- 步骤 2: 无梯度采样 (适配新接口) ---
                with torch.no_grad():
                    # 关键修改：解构返回的 ids 和软特征
                    sampled_ids_list, soft_features_list = raw_model.mbr_sampling(
                        batch["text_onehot"], batch["text_mask"], batch["error_stats"],
                        max_len=model_config.max_len, temperature=mbr_tau, k=mbr_k
                    )
                    # 计算常数风险矩阵 [B, K]
                    r_matrix = compute_batch_penalty(
                        sampled_ids_list, batch["text_ids"], batch["error_stats"], calc
                    ).to(model_engine.device)

                # --- 步骤 3: 并行化有梯度对数概率回算 ---
                # 关键修改：同时传入 ids 和采样时记录的 soft_features 以消灭 for 循环
                log_probs_matrix = raw_model.mbr_log_probs(
                    batch["text_onehot"], 
                    batch["text_mask"], 
                    batch["error_stats"], 
                    sampled_ids_list,
                    soft_features_list
                ) # [B, K]

                # --- 步骤 4: 计算期望风险损失 ---
                with torch.no_grad():
                    lp_max = log_probs_matrix.max(dim=1, keepdim=True)[0]
                    lp_min = log_probs_matrix.min(dim=1, keepdim=True)[0]
                    lp_range = lp_max - lp_min + 1e-6
                        
                    # 强行映射到 [0, 1] 比例，然后再放大到合适倍数
                normalized_lp = (log_probs_matrix - lp_max) / lp_range * 1.5 # 可调，越大越尖锐
                q_matrix = torch.softmax(normalized_lp, dim=-1)
                per_sample_mbr_loss = (q_matrix * r_matrix).sum(dim=1)
                loss_mbr = per_sample_mbr_loss.mean()

                # === MBR 逻辑正确性检查代码 ===
                debug_interval = log_config.get("debug_interval", 100)
                if rank == 0 and global_step % debug_interval == 0:
                    with torch.no_grad():
                        sample_idx = 0 
                        sample_k_paths = []
                        for k_idx in range(mbr_k):
                            path_ids = sampled_ids_list[k_idx][sample_idx]
                            path_tuple = tuple(path_ids[path_ids != 0].cpu().tolist())
                            sample_k_paths.append(path_tuple)
                            
                        unique_paths = len(set(sample_k_paths))
                        r_sample = r_matrix[sample_idx]
                        q_sample = q_matrix[sample_idx]
                        
                        print(f"\n{'='*20} MBR Debug (Step: {global_step}) {'='*20}")
                        print(f"Sample 0 Diversity: {unique_paths}/{mbr_k} unique paths")
                        
                        for k_idx in range(mbr_k):
                            path_preview = list(sample_k_paths[k_idx][:10])
                            print(f"  Path {k_idx}: Penalty={r_sample[k_idx]:.4f}, Prob(q)={q_sample[k_idx]:.4f} | IDs: {path_preview}...")
                        
                        best_k = torch.argmin(r_sample).item()
                        worst_k = torch.argmax(r_sample).item()
                        print(f"  --> Comparison: Best(Pnl:{r_sample[best_k]:.3f}/Prob:{q_sample[best_k]:.3f}) vs Worst(Pnl:{r_sample[worst_k]:.3f}/Prob:{q_sample[worst_k]:.3f})")
                        
                        entropy = -(q_sample * torch.log(q_sample + 1e-9)).sum()
                        print(f"  --> Softmax Entropy: {entropy:.4f}")
                        print(f"{'='*55}\n")
                # ============================================
                
                # loss = loss_mbr # 暂时微调使用纯penalty
                loss = loss_mbr
                # loss = loss_ce + mbr_lambda * loss_mbr
                penalty_log = r_matrix.mean()
            else:
                loss = loss_ce
                penalty_log = torch.tensor(0.0)

            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1

            if rank == 0:
                cur_lr = model_engine.get_lr()[0]
                pbar.set_postfix(
                    ce_loss=f"{loss_ce.item():.4f}", 
                    pnl=f"{penalty_log.item():.4f}" if use_mbr else "OFF",
                    acc=f"{acc.item():.4f}", 
                    lr=f"{cur_lr:.2e}")
                pbar.update(1)
                
                if global_step % log_config.log_interval == 0:
                    step_logger.info(
                        f"Epoch: {epoch+1}, Step: {step}, Global: {global_step}, "
                        f"Loss: {loss.item():.4f}, Penalty: {penalty_log.item():.4f}, CE_Loss: {loss_ce.item():.4f}," 
                        f"Acc: {acc.item():.4f}, LR: {cur_lr:.2e}"
                    )

            if train_config.run_validation and global_step % train_config.validation_interval == 0:
                if rank == 0 and pbar: pbar.close()
                
                val_loss = evaluate(model_engine, dev_dataloader, rank, model_config, calc)
                model_engine.train()
                
                if rank == 0:
                    remaining = len(train_dataloader) - (step + 1)
                    if remaining > 0:
                        pbar = tqdm(total=remaining, desc=f"Epoch {epoch+1} (Cont)", dynamic_ncols=True)

                if train_config.save_model:
                    save_path = os.path.join(train_config.output_dir, "checkpoints")
                    tag = f"step_{global_step}"
                    
                    if rank == 0:
                        logger.info(f"Saving DeepSpeed shards to {save_path} tag {tag}")
                    
                    model_engine.save_checkpoint(save_path, tag)
                    dist.barrier()

                    if rank == 0:
                        checkpoint_dir = os.path.join(save_path, tag)
                        output_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
                        try:
                            convert_zero_checkpoint_to_fp32_state_dict(save_path, output_file)
                            logger.info("Conversion successful!")
                        except Exception as e:
                            logger.error(f"Failed to convert checkpoint: {e}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if rank == 0:
                            logger.info(f"New best validation loss: {best_loss:.4f}")

        if rank == 0 and pbar:
            pbar.close()

    if rank == 0:
        logger.info("Training Finished.")

def evaluate(model_engine, dataloader, rank, model_config, calculator):
    """
    修改后的验证函数：
    1. 遵循非贪婪采样逻辑，使用与训练一致的采样温度和软反馈链条。
    2. 局部累加指标，循环外执行单次分布式汇总，彻底解决 HCCL 报错。
    """
    if dataloader is None:
        return float('inf')
        
    model_engine.eval()
    
    # 局部累加器（初始化为 Python 标量）
    local_sum_combined = 0.0
    local_sum_ce = 0.0
    local_sum_penalty = 0.0
    local_sum_acc = 0.0
    local_steps = 0
    
    mbr_lambda = getattr(model_config, "mbr_lambda", 2.0)
    # 验证阶段使用与训练一致的非贪婪采样温度
    sampling_temp = getattr(model_config, "mbr_tau", 0.8)
    max_len = getattr(model_config, "max_len", 160)
    
    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Validating (MBR Soft-Inference)", 
                    colour="green", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, model_engine.device)
            
            # 1. 计算基础 Teacher Forcing 指标
            outputs, acc = model_engine(
                text_onehot=batch["text_onehot"],
                text_mask=batch["text_mask"],
                target_psd=batch["target_psd"],
                target_lengths=batch["target_lengths"],
                error_stats=batch["error_stats"]
            )
            loss_ce = outputs.loss
            
            # 2. 计算基于非贪婪软反馈的 Penalty
            raw_model = model_engine.module.simulator if hasattr(model_engine.module, "simulator") else model_engine.module
            
            # 关键修改：使用非贪婪温度采样，模拟真实 soft-AR 推理
            sampled_ids, _ = raw_model.mbr_sampling(
                batch["text_onehot"], 
                batch["text_mask"], 
                batch["error_stats"],
                max_len=max_len, 
                temperature=sampling_temp, 
                k=1 # 验证时每条数据取一路采样即可，但必须是非贪婪过程
            )
            
            # 计算当前批次平均偏差
            penalty_matrix = compute_batch_penalty(
                sampled_ids, 
                batch["text_ids"], 
                batch["error_stats"], 
                calculator
            )
            avg_penalty = penalty_matrix.mean().to(loss_ce.device)

            # 3. 本地累加指标
            local_sum_combined += (loss_ce + mbr_lambda * avg_penalty).item()
            local_sum_ce += loss_ce.item()
            local_sum_penalty += avg_penalty.item()
            local_sum_acc += acc.item() if isinstance(acc, torch.Tensor) else acc
            local_steps += 1
            
            if rank == 0:
                pbar.update(1)
    
    if rank == 0 and pbar:
        pbar.close()

    # --- 关键修改：分布式汇总 ---
    if dist.is_initialized():
        # 封装到张量，统一转移到 NPU，强制 float32 避免类型报错
        metrics = torch.tensor([
            local_sum_combined, local_sum_ce, local_sum_penalty, local_sum_acc, float(local_steps)
        ], device=model_engine.device, dtype=torch.float32)
        
        # 确保所有 Rank 到达后再同步
        dist.barrier() 
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        # 解包全局汇总值
        sum_comb, sum_ce, sum_p, sum_acc, global_steps = metrics.tolist()
    else:
        sum_comb, sum_ce, sum_p, sum_acc, global_steps = \
            local_sum_combined, local_sum_ce, local_sum_penalty, local_sum_acc, float(local_steps)

    avg_loss_combined = sum_comb / (global_steps + 1e-6)
    
    if rank == 0:
        logger.info(f"--- MBR Validation Result (Temp={sampling_temp}) ---")
        logger.info(f"Combined Loss: {avg_loss_combined:.4f}")
        logger.info(f"Base CE: {sum_ce/(global_steps+1e-6):.4f} | "
                    f"Avg Penalty: {sum_p/(global_steps+1e-6):.4f} | "
                    f"Acc: {sum_acc/(global_steps+1e-6):.4f}")
        
    return avg_loss_combined

def to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch

def setup_logging(log_config, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log"), mode='a')
        ]
    )

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch_npu.npu.manual_seed_all(seed)
    except:
        pass

if __name__ == "__main__":
    main_hydra()