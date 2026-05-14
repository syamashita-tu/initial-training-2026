a = [4, 8, 3, 4, 1]
a1 = a.copy()

a1 = [a1[i] % 2 for i in range(len(a1))]
print(a1)

a2 = sum(a1)
print(a2)

a3 = [a[i] for i in range(len(a1)) if a1[i] == 1]
print(a3)