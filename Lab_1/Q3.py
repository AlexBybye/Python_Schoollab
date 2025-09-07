low = 123
high = 456789
total = 0

for n in range(low, high + 1):   # 包含 high!!!
    total += n

print('Sum from', low, 'to', high, '=', total)