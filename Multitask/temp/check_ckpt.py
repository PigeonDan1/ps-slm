import torch
import os

# 修改为你的 checkpoint 路径
CKPT_PATH = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/exp/company/libri_lora_realctc-20260215-1002/ps-slm_epoch_4_step_1000/pytorch_model.bin"

def inspect():
    if not os.path.exists(CKPT_PATH):
        print(f"错误: 找不到文件 {CKPT_PATH}")
        return

    print(f"正在读取: {CKPT_PATH}")
    # 适配 PyTorch 2.6 的安全加载
    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    
    # 如果是 DeepSpeed 保存的，可能嵌套在 'module' 或 'state_dict' 键下
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    
    keys = list(checkpoint.keys())
    total_keys = len(keys)
    
    print(f"\n" + "="*50)
    print(f"统计汇总:")
    print(f"总键数 (Total Keys): {total_keys}")
    print(f"="*50)

    # 1. 检查是否存在 module. 前缀
    has_module_prefix = any(k.startswith("module.") for k in keys)
    print(f"包含 'module.' 前缀: {has_module_prefix}")

    # 2. 检查组件分布
    lora_keys = [k for k in keys if "lora" in k]
    projector_keys = [k for k in keys if "encoder_projector" in k]
    encoder_keys = [k for k in keys if "encoder.encoder" in k]
    llm_keys = [k for k in keys if "llm." in k]

    print(f"\n组件分布:")
    print(f"- LoRA 相关键数: {len(lora_keys)}")
    print(f"- Projector (Adapter) 相关键数: {len(projector_keys)}")
    print(f"- Encoder 相关键数: {len(encoder_keys)}")
    print(f"- LLM 基础权重键数: {len(llm_keys)}")

    # 3. 打印前 15 个键名样例，观察层级结构
    print(f"\n键名样例 (前 15 个):")
    for k in keys[:15]:
        print(f"  {k}")

    # 4. 如果有 LoRA，打印一个 LoRA 键名观察
    if lora_keys:
        print(f"\nLoRA 键名完整示例:")
        print(f"  {lora_keys[0]}")

    print(f"\n诊断建议:")
    if not lora_keys:
        print("!!! [警告] Checkpoint 中未检测到 LoRA 权重，请检查一阶段训练是否开启了 PEFT。")
    if has_module_prefix:
        print(">>> 权重包含 'module.' 前缀。如果训练脚本在加载时模型没有该前缀，会导致加载失败。")
    else:
        print(">>> 权重不包含 'module.' 前缀。如果训练脚本使用了 DeepSpeed 封装，可能需要手动补齐前缀。")

if __name__ == "__main__":
    inspect()