#!/usr/bin/env python3
# clean_punct.py
import sys
import pathlib
import string
import unicodedata

# 中英文字符标点 + 反斜杠
PUNCT_SET = set(string.punctuation) | {
    '，', '。', '！', '？', '：', '；', '、', '（', '）',
    '“', '”', '‘', '’', '【', '】', '《', '》', '——', '…',
    '\\'  # 反斜杠
}

def is_valid_char(ch):
    """
    判断字符是否为合法字符：
    - 可打印
    - unicodedata.name 不报错
    - 不是标点
    """
    try:
        unicodedata.name(ch)
    except ValueError:
        return False
    return ch.isprintable() and ch not in PUNCT_SET

def strip_all_punct(path):
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        print(f'文件不存在: {path}')
        sys.exit(1)

    lines = path.read_text(encoding='utf-8').splitlines()
    cleaned_lines = []

    for line in lines:
        if '\t' not in line:
            cleaned_lines.append(line)
            continue
        key, text = line.split('\t', 1)
        # 删除所有标点和异常字符
        text = ''.join(ch for ch in text if is_valid_char(ch))
        cleaned_lines.append(f'{key}\t{text}')

    path.write_text('\n'.join(cleaned_lines) + '\n', encoding='utf-8')
    print('已清除所有标点和异常字符并保存。')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('用法: python clean_marks.py <预测文件路径>')
        sys.exit(1)
    strip_all_punct(sys.argv[1])