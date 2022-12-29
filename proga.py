n = 1000
osn = 3 # int
s = '' # string 
while n > 0:
    ost = n % osn
    s = s + str(ost)
    n = n // osn
    #КОНКАТЕНАЦИЯ == СЛОЖЕНИЕ НЕСКОЛЬКИХ СТРОК
print('AAAA'.join(reversed(s)))
print(s[::-1])