import json

# 定义输入和输出路径
input_path = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/ckpt_text/decode_en2zh_st_test_st_gt"
output_path = input_path + ".jsonl"

def convert_to_jsonl(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as f_in, \
         open(out_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # 按空白符分割 key 和 label
            parts = line.split(None, 1)
            if len(parts) == 2:
                # 构造符合仓库要求的字典格式
                # 将 task 设为 "ser" 以便直接运行评估脚本
                item = {
                    "key": parts[0],
                    "task": "s2tt", 
                    "target": parts[1]
                }
                # 写入 jsonl
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    convert_to_jsonl(input_path, output_path)
    print(f"转换完成，文件保存在: {output_path}")