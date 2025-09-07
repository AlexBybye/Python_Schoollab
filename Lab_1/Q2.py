def level_diagnoise(score:int)->str:
    if score >= 90 and score<=100:
        return 'A'
    elif score >= 80 and score<90:
        return 'B'
    elif score >= 60 and score<80:
        return 'C'
    elif score >= 0 and score<60:
        return 'D'
    else:
        return 'False'
while True:
    tempt =int(input("Time to enter:(Quit with -1)")) 
    if tempt == -1:
        break
    else:
        score=int(tempt)
        print(level_diagnoise(score))
        continue