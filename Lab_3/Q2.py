# Q2.py

from collections import Counter

def valid_number():
    numbers = []

    def is_valid(n):
        s = str(n)
        if len(s) != 5:
            return False
        # 检查是否只包含0-5这几个数字
        for digit in s:
            if digit not in "012345":
                return False

        counts = Counter(s)
        for count in counts.values():
            if count > 2:
                return False
        return True

    for i in range(10000, 60000):
        if is_valid(i):
            numbers.append(i)

    return numbers
print(valid_number())