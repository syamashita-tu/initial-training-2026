from PIL import Image
import numpy as np

img = Image.open("syamashita-tu/burst_denoising/data/raw/u10_Ship_2K.tif")

img.save("syamashita-tu/burst_denoising/data/inputs/u10_Ship_2K.png")