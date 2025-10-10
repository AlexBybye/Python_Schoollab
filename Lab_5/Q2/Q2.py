from collections import Counter
import re


def count_letter():
    with open("./Q2.txt", "r") as f:
        text = f.read()

    # 将文本转换为小写
    text_lower = text.lower()

    # 正则表达式提取所有字母
    letters_list = re.findall('[a-z]', text_lower)

    # 使用 Counter 统计频率
    letter_counts = Counter(letters_list)

    return dict(letter_counts.items())


print(count_letter())