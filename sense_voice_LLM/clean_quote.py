#!/usr/bin/env python3
# clean_quote.py
import sys
import pathlib

def strip_single_quotes(path):
    """
    删除预测文件中所有单引号（包括英文和中文单引号变体），原地覆盖。
    """
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        print(f'文件不存在: {path}')
        sys.exit(1)

    # 读入所有行
    lines = path.read_text(encoding='utf-8').splitlines()

    # 需要删除的字符集合
    quote_chars = {"'", "‘", "’"}

    cleaned_lines = []
    for line in lines:
        # 仅保留 tab 左侧的 key 与右侧的文本
        if '\t' not in line:
            cleaned_lines.append(line)           # 异常行原样保留
            continue
        key, text = line.split('\t', 1)
        # 删除所有单引号
        text = ''.join(ch for ch in text if ch not in quote_chars)
        cleaned_lines.append(f'{key}\t{text}')

    # 写回文件
    path.write_text('\n'.join(cleaned_lines) + '\n', encoding='utf-8')
    print('已清除所有单引号并保存。')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('用法: python clean_quote.py <预测文件路径>')
        sys.exit(1)
    strip_single_quotes(sys.argv[1])