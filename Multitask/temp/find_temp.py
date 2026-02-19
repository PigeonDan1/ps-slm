import random

def sample_jsonl(src_path, dst_path, k=50000):
    """
    从源 JSONL 文件中随机抽取 k 条数据并保存到新文件。
    """
    # 统计总行数以确定采样索引
    with open(src_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    # 随机选取索引并排序（升序处理文件流更高效）
    sampled_indices = set(random.sample(range(line_count), min(k, line_count)))

    with open(src_path, 'r', encoding='utf-8') as f_in:
        with open(dst_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i in sampled_indices:
                    f_out.write(line)

# 执行抽样
src = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train/multitask.jsonl"
dst = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/temp_jsonl/libri_origin_50k.jsonl"

sample_jsonl(src, dst)