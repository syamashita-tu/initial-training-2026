#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Burst画像の多フレームDenoising課題用プログラム

基礎レベル
1. 入力画像を縦横1/2に縮小して原画像 x を作成
2. 平均0，分散0.1の独立ガウスノイズを加え，15枚のburst画像を生成
3. K = 3,5,7,9,11,13,15 について算術平均ベースラインを評価
4. 改良手法として
   - 最大値・最小値除外平均（trimmed mean）
   - 中央値（median）
   を実装
5. PSNR，SSIMを計算し，CSV・グラフ・比較画像を保存

発展レベル（--advanced 指定時）
6. ごま塩ノイズ，ラプラスノイズでも同様に比較

必要パッケージ:
    pip install numpy pillow matplotlib scipy scikit-image
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter


K_VALUES = [3, 5, 7, 9, 11, 13, 15]
METHODS = ["baseline_mean", "trimmed_mean", "median", "mean_gaussian_filter"]


def load_original_image(path: str) -> np.ndarray:
    """画像をRGBで読み込み，縦横1/2に縮小し，[0,1]のfloat32配列にする．"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_size = (max(1, w // 2), max(1, h // 2))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)


def save_float_image(img: np.ndarray, path: Path) -> None:
    """[0,1]画像をPNGとして保存する．"""
    img8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(path)


def generate_gaussian_burst(
    x: np.ndarray,
    num_frames: int,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    y_k = clip(x + epsilon_k, 0, 1)
    epsilon_k ~ N(0, variance)
    """
    std = np.sqrt(variance)
    noise = rng.normal(
        loc=0.0,
        scale=std,
        size=(num_frames,) + x.shape
    ).astype(np.float32)

    burst = np.clip(x[None, ...] + noise, 0.0, 1.0)
    return burst.astype(np.float32)


def generate_laplace_burst(
    x: np.ndarray,
    num_frames: int,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    ラプラス分布の分散をGaussianと同じ variance に合わせる．
    Laplace(0,b) の分散は 2b^2 なので b = sqrt(variance/2)
    """
    scale = np.sqrt(variance / 2.0)
    noise = rng.laplace(
        loc=0.0,
        scale=scale,
        size=(num_frames,) + x.shape
    ).astype(np.float32)

    burst = np.clip(x[None, ...] + noise, 0.0, 1.0)
    return burst.astype(np.float32)


def generate_salt_pepper_burst(
    x: np.ndarray,
    num_frames: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    各画素・各チャネルについて確率 probability でインパルスノイズを付加する．
    その半分を0（pepper），半分を1（salt）にする．
    """
    burst = np.repeat(x[None, ...], num_frames, axis=0).copy()

    r = rng.random(size=burst.shape)
    pepper = r < (probability / 2.0)
    salt = (r >= probability / 2.0) & (r < probability)

    burst[pepper] = 0.0
    burst[salt] = 1.0
    return burst.astype(np.float32)


def baseline_mean(frames: np.ndarray) -> np.ndarray:
    """課題指定の算術平均ベースライン．"""
    return np.clip(np.mean(frames, axis=0), 0.0, 1.0)


def trimmed_mean(frames: np.ndarray) -> np.ndarray:
    """
    各画素・各RGBチャネルについて最大値と最小値を1個ずつ除外して平均する．
    K >= 3 を前提とする．
    """
    if frames.shape[0] < 3:
        raise ValueError("trimmed_meanには3枚以上のフレームが必要です")

    sorted_frames = np.sort(frames, axis=0)
    return np.clip(np.mean(sorted_frames[1:-1], axis=0), 0.0, 1.0)


def median_fusion(frames: np.ndarray) -> np.ndarray:
    """各画素・各RGBチャネルについて中央値を取る．"""
    return np.clip(np.median(frames, axis=0), 0.0, 1.0)


def mean_gaussian_filter(
    frames: np.ndarray,
    spatial_sigma: float = 0.8,
) -> np.ndarray:
    """
    burstを算術平均した後，空間方向だけGaussianフィルタを適用する．
    RGBチャネル方向には平滑化しない．
    """
    mean_img = baseline_mean(frames)
    filtered = gaussian_filter(
        mean_img,
        sigma=(spatial_sigma, spatial_sigma, 0.0),
        mode="reflect",
    )
    return np.clip(filtered, 0.0, 1.0)


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """RGB全チャネルを含むMSE．"""
    return float(np.mean((x.astype(np.float64) - x_hat.astype(np.float64)) ** 2))


def psnr(x: np.ndarray, x_hat: np.ndarray) -> float:
    """画素値範囲[0,1]に対するPSNR．"""
    error = mse(x, x_hat)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / error))


def ssim(x: np.ndarray, x_hat: np.ndarray) -> float:
    """RGB画像のSSIM．"""
    return float(
        structural_similarity(
            x,
            x_hat,
            channel_axis=2,
            data_range=1.0
        )
    )


def reconstruct(
    frames: np.ndarray,
    method: str,
    filter_sigma: float,
) -> np.ndarray:
    if method == "baseline_mean":
        return baseline_mean(frames)
    if method == "trimmed_mean":
        return trimmed_mean(frames)
    if method == "median":
        return median_fusion(frames)
    if method == "mean_gaussian_filter":
        return mean_gaussian_filter(frames, spatial_sigma=filter_sigma)
    raise ValueError(f"Unknown method: {method}")


def evaluate_burst(
    x: np.ndarray,
    burst: np.ndarray,
    filter_sigma: float,
) -> list[dict]:
    """Kごと・手法ごとにPSNRとSSIMを計算する．"""
    results = []

    for k in K_VALUES:
        frames = burst[:k]

        for method in METHODS:
            x_hat = reconstruct(frames, method, filter_sigma)
            results.append(
                {
                    "K": k,
                    "method": method,
                    "PSNR": psnr(x, x_hat),
                    "SSIM": ssim(x, x_hat),
                }
            )

    return results


def save_metrics_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "method", "PSNR", "SSIM"])
        writer.writeheader()
        writer.writerows(results)


def print_metrics(results: list[dict], title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'K':>3}  {'method':<16}  {'PSNR [dB]':>12}  {'SSIM':>10}")
    print("-" * 76)

    for row in results:
        print(
            f"{row['K']:>3}  "
            f"{row['method']:<16}  "
            f"{row['PSNR']:>12.4f}  "
            f"{row['SSIM']:>10.6f}"
        )


def plot_metric(
    results: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(7, 5))

    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        ks = [r["K"] for r in rows]
        vals = [r[metric] for r in rows]
        plt.plot(ks, vals, marker="o", label=method)

    plt.xlabel("Number of frames K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(K_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_comparison_figure(
    x: np.ndarray,
    burst: np.ndarray,
    noise_name: str,
    path: Path,
    filter_sigma: float,
) -> None:
    """K=15で原画像，1枚目，ベースライン，改良3手法を並べる．"""
    k = 15
    frames = burst[:k]

    images = [
        x,
        burst[0],
        baseline_mean(frames),
        trimmed_mean(frames),
        median_fusion(frames),
        mean_gaussian_filter(frames, spatial_sigma=filter_sigma),
    ]

    titles = [
        "Original",
        "Burst frame 1",
        f"Mean\nPSNR={psnr(x, images[2]):.2f} dB",
        f"Trimmed mean\nPSNR={psnr(x, images[3]):.2f} dB",
        f"Median\nPSNR={psnr(x, images[4]):.2f} dB",
        f"Mean + Gaussian\nPSNR={psnr(x, images[5]):.2f} dB",
    ]

    fig, axes = plt.subplots(1, 6, figsize=(21, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"{noise_name}: comparison at K=15")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_burst_images(burst: np.ndarray, out_dir: Path, prefix: str) -> None:
    burst_dir = out_dir / f"{prefix}_bursts"
    burst_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(burst, start=1):
        save_float_image(frame, burst_dir / f"burst_{i:02d}.png")


def best_method_at_k15(results: list[dict]) -> dict:
    rows = [r for r in results if r["K"] == 15]
    return max(rows, key=lambda r: r["PSNR"])


def run_one_noise(
    x: np.ndarray,
    burst: np.ndarray,
    noise_name: str,
    out_dir: Path,
    filter_sigma: float,
) -> None:
    results = evaluate_burst(x, burst, filter_sigma)

    print_metrics(results, f"{noise_name} noise")
    best = best_method_at_k15(results)

    print()
    print(
        f"[{noise_name}] K=15でPSNR最大: "
        f"{best['method']}  "
        f"PSNR={best['PSNR']:.4f} dB, "
        f"SSIM={best['SSIM']:.6f}"
    )

    save_metrics_csv(results, out_dir / f"metrics_{noise_name}.csv")

    plot_metric(
        results,
        metric="PSNR",
        ylabel="PSNR [dB]",
        title=f"PSNR vs K ({noise_name})",
        path=out_dir / f"psnr_{noise_name}.png",
    )

    plot_metric(
        results,
        metric="SSIM",
        ylabel="SSIM",
        title=f"SSIM vs K ({noise_name})",
        path=out_dir / f"ssim_{noise_name}.png",
    )

    save_comparison_figure(
        x,
        burst,
        noise_name,
        out_dir / f"comparison_{noise_name}_K15.png",
        filter_sigma,
    )

    save_burst_images(burst, out_dir, noise_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burst画像の多フレームDenoising課題"
    )
    parser.add_argument(
        "--input",
        default="./syamashita-tu/burst_denoising/data/inputs",
        help="入力画像ディレクトリ"
    )
    parser.add_argument(
        "--output",
        default="./syamashita-tu/burst_denoising/data/outputs",
        help="結果保存ディレクトリ"
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=0.1,
        help="Gaussian/Laplaceノイズの分散（既定値: 0.1）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数seed（既定値: 42）"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="発展課題のごま塩・Laplaceノイズも実行する"
    )
    parser.add_argument(
        "--sp-prob",
        type=float,
        default=0.1,
        help="ごま塩ノイズの発生確率（既定値: 0.1）"
    )
    parser.add_argument(
        "--filter-sigma",
        type=float,
        default=0.8,
        help="改良法のGaussianフィルタ標準偏差（既定値: 0.8）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.variance < 0:
        raise ValueError("varianceは0以上で指定してください")
    if not (0.0 <= args.sp_prob <= 1.0):
        raise ValueError("sp-probは0以上1以下で指定してください")
    if args.filter_sigma < 0:
        raise ValueError("filter-sigmaは0以上で指定してください")

    input_dir = Path(args.input)
    output_root = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if not image_paths:
        raise FileNotFoundError(
            f"入力画像が見つかりません: {input_dir}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input directory : {input_dir.resolve()}")
    print(f"Output directory: {output_root.resolve()}")
    print(f"Number of images: {len(image_paths)}")

    for image_index, image_path in enumerate(image_paths, start=1):
        print()
        print("#" * 80)
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")
        print("#" * 80)

        # 画像ごとに出力フォルダを分ける
        out_dir = output_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 課題1: 原画像
        x = load_original_image(str(image_path))
        save_float_image(x, out_dir / "original_half.png")

        print(f"Original shape: {x.shape}")
        print(f"Gaussian variance: {args.variance}")
        print(f"Gaussian std: {np.sqrt(args.variance):.8f}")

        # 画像ごとに再現可能な乱数系列を生成
        seed_seq = np.random.SeedSequence([args.seed, image_index])
        child_seeds = seed_seq.spawn(3)

        # 課題1〜4: Gaussian
        rng_gaussian = np.random.default_rng(child_seeds[0])
        gaussian_burst = generate_gaussian_burst(
            x=x,
            num_frames=15,
            variance=args.variance,
            rng=rng_gaussian,
        )
        run_one_noise(
            x=x,
            burst=gaussian_burst,
            noise_name="gaussian",
            out_dir=out_dir,
            filter_sigma=args.filter_sigma,
        )

        # 課題5: 発展レベル
        if args.advanced:
            rng_sp = np.random.default_rng(child_seeds[1])
            sp_burst = generate_salt_pepper_burst(
                x=x,
                num_frames=15,
                probability=args.sp_prob,
                rng=rng_sp,
            )
            run_one_noise(
                x=x,
                burst=sp_burst,
                noise_name="salt_pepper",
                out_dir=out_dir,
                filter_sigma=args.filter_sigma,
            )

            rng_laplace = np.random.default_rng(child_seeds[2])
            laplace_burst = generate_laplace_burst(
                x=x,
                num_frames=15,
                variance=args.variance,
                rng=rng_laplace,
            )
            run_one_noise(
                x=x,
                burst=laplace_burst,
                noise_name="laplace",
                out_dir=out_dir,
                filter_sigma=args.filter_sigma,
            )

        print(f"結果保存先: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
