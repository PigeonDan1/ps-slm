import json
import os

# ================= 配置区域 =================
# 原始输入文件路径
input_raw_jsonl = "/aistor/sjtu/hpc_stor01/home/yangyi/data/MMSU/multitask.jsonl"
# 目标保存目录
output_dir = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/ckpt_text"
# 目标保存文件名
output_filename = "slu_standard_gt.jsonl"
# ===========================================

def generate_slu_gt():
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_filename)
    
    processed_count = 0
    with open(input_raw_jsonl, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            item = json.loads(line)
            
            # 构建标准的 SLU 评估条目
            # 1. 强制任务类型为 slu 以触发 evaluator.py 中的 slu_eval 逻辑
            # 2. 确保包含 prompt 字段，供 process_prediction.py 解析 A/B/C/D 选项
            new_item = {
                "key": item['key'],
                "task": "slu",
                "target": item.get('target') or item.get('GT'),
                "prompt": item.get('prompt', "")
            }
            
            f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"成功转换 {processed_count} 条数据")
    print(f"标准 GT 已保存至: {output_path}")

if __name__ == "__main__":
    generate_slu_gt()