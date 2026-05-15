from pathlib import Path

folder = Path("syamashita-tu/chapter05")
n_list = list(range(1, 11))  # 任意の数を指定

folder.mkdir(parents=True, exist_ok=True)

for n in n_list:
    if n < 10:
        file_path = folder / f"q0{n}.ipynb"
    else:
        file_path = folder / f"q{n}.ipynb"
    file_path.touch(exist_ok=True)