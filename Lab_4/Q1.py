def cycle(iterable):
    """
    生成器：接受一个可迭代对象，并无限重复地生成其中的元素。

    参数:
    iterable: 一个可迭代对象 (例如，字符串, 列表, 元组)。
    """
    saved_elements = tuple(iterable)

    while True:
        for element in saved_elements:
            yield element


# 测试代码 (来自 Q1.py)
str_data = 'abcd'
g = cycle(str_data)

for i in range(99):
    "前99个"
    try:
        print(next(g))
    except StopIteration:
        break