from pathlib import Path

folder = Path("syamashita-tu/chapter04")
n_list = list(range(1, 11))  # 任意の数を指定

folder.mkdir(parents=True, exist_ok=True)

for n in n_list:
    file_path = folder / f"q0{n}.ipynb"
    file_path.touch(exist_ok=True)