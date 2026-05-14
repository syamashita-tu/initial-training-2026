a = 1.2 + 3.8
b = 10 // 100
c = 1 >= 0
d = 'Hello World' == 'Hello World'
e = not 'Chainer' != 'Tutorial'
f = all([True, True, False]) 
g = any([True, True, False])
h = abs(-3) 
i = 2 // 1

list = [a,b,c,d,e,f,g,h,i]

for j in range(len(list)):
    print(j)
    print(list[j])
    print(type(list[j]))