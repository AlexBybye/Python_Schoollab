# 在下面自行定义strip函数
def strip(s: str, side: str = "both", blanks: list = None) -> str:

    if blanks is None:
        blanks = ["\t", "\n", " "]

    if side == "left" or side == "both":
        start_index = 0
        while start_index < len(s) and s[start_index] in blanks:
            start_index += 1
        s = s[start_index:]

    if side == "right" or side == "both":
        end_index = len(s) - 1
        while end_index >= 0 and s[end_index] in blanks:
            end_index -= 1
        s = s[:end_index + 1]

    return s

print("\"" + strip("   abc   ") + "\"")
print("\"" + strip("   abc   ", "both") + "\"")
print("\"" + strip("   abc   ", "left") + "\"")
print("\"" + strip("   abc   ", side="right") + "\"")
print("\"" + strip("aaadefzzz  ", blanks=["a", "z", " "]) + "\"")