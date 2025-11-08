#!/usr/bin/env python3
# clean_punct.py
import sys
import pathlib
import string
import unicodedata

# Chinese and English characters punctuation + backslash
PUNCT_SET = set(string.punctuation) | {
    '，', '。', '！', '？', '：', '；', '、', '（', '）',
    '“', '”', '‘', '’', '【', '】', '《', '》', '——', '…',
    '\\' 
}

def is_valid_char(ch):
    """
    Determine whether the character is a valid character:
    - Printable
    - unicodedata.name 
    - Not punctuation
    """
    try:
        unicodedata.name(ch)
    except ValueError:
        return False
    return ch.isprintable() and ch not in PUNCT_SET

def strip_all_punct(path):
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        print(f'The file does not exist: {path}')
        sys.exit(1)

    lines = path.read_text(encoding='utf-8').splitlines()
    cleaned_lines = []

    for line in lines:
        if '\t' not in line:
            cleaned_lines.append(line)
            continue
        key, text = line.split('\t', 1)
        # Remove all punctuation and abnormal characters
        text = ''.join(ch for ch in text if is_valid_char(ch))
        cleaned_lines.append(f'{key}\t{text}')

    path.write_text('\n'.join(cleaned_lines) + '\n', encoding='utf-8')
    print('All punctuation and abnormal characters have been cleared and saved.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python clean_marks.py <path_to_prediction_file>')
        sys.exit(1)
    strip_all_punct(sys.argv[1])
