prime = []

for p in range(2, 101):
    is_prime = True

    for k in range(2, p):
        if p % k == 0:
            is_prime = False
            break

    if is_prime:
        prime.append(p)

print(prime)