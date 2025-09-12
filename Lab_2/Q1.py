def strip(str):
    l = 0
    r = len(str)

    # Move 'l' past any leading whitespace
    while l < r and str[l].isspace():
        l += 1

    # Move 'r' back past any trailing whitespace
    while l < r and str[r - 1].isspace():
        r -= 1

    return str[l:r]
print(strip('\n\t a\n b\tc\n ancedote'))