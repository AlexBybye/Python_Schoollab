def count_letter():
    with open("./Q2.txt", "r") as f:
        text = f.read()
    counts = {}

    # 请补全代码
    for char in text:
        if 'a' <= char.lower() <= 'z':
            letter = char.lower()
            counts[letter] = counts.get(letter, 0) + 1

    return counts


print(count_letter())