a = [4, 8, 3, 4, 1]

# 先頭の要素を取り除く
b = a.copy()
b.pop(0)
print(b)  # [8, 3, 4, 1]

# 末尾の要素を取り除く
c = a.copy()
c.pop(-1)
print(c)  # [4, 8, 3, 4]

# 末尾に 100 を追加する
d = a.copy()
d.append(100)
print(d)  # [4, 8, 3, 4, 1, 100]