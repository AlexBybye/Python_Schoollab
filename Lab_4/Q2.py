import time
import functools


def timeit(func):
    """
    装饰器：用于计算并输出被装饰函数的运行时间。
    """

    # 使用 functools.wraps 来保留被装饰函数的元数据 (如 __name__, __doc__)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # 执行被装饰的函数，并捕获其返回值
        result = func(*args, **kwargs)
        end_time = time.time()

        # 计算运行时间
        run_time = end_time - start_time

        # 输出运行时间
        print(f"函数执行时间：{run_time}秒")

        # 返回被装饰函数的执行结果
        return result

    # 装饰器返回 wrapper 函数
    return wrapper


# 使用样例：
@timeit
def func(num):
    # 这是一个计算从 0 到 num-1 整数和的函数
    return sum(range(num))


# 调用被装饰的函数，装饰器会在其执行前后自动计时并打印时间
# print() 语句将打印 func 的返回值，即 sum(range(2000000)) 的结果。
print(func(2000000))