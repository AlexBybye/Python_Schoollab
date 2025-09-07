numbers = [33, 51, 56, 44, 82, 15]
even = []
odd = []

while numbers:         
    number = numbers.pop()
    
    if number % 2 == 0:  
        even.append(number)
    else:
        odd.append(number)

print('even:', even)
print('odd:', odd)