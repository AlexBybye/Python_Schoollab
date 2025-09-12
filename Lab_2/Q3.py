def unique_sorted_list():
    numbers_str = input("请输入一串整数（以空格分隔）：").split()
    numbers = [int(n) for n in numbers_str]  # Convert input strings to integers!!

    result = []

    # 请补全代码
    unique_numbers = sorted(list(set(numbers)))
    result = unique_numbers

    return result


print(unique_sorted_list())