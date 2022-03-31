#!/usr/bin/env python3
"""
read a geotiff file, read its metadata, and visualize a point plotted on the image
"""

import rasterio
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from rasterio.plot import show


def main():
  fname = "2019-02-25-00_00_2019-02-25-23_59_Sentinel-2_L2A_True_color.tiff"

  # https://rasterio.readthedocs.io/en/latest/topics/plotting.html

  with rasterio.open(fname) as src:
    print(src.meta)

    point = (612, 267)

    figure, axes = plt.subplots()
    axes.add_artist(plt.Circle(point, radius=3, color="red", fill=False))
    plt.imshow(src.read(1), cmap='pink')
    #plt.imshow(src.read(1))


    plt.show()

    #show(src.read(), transform=src.transform)



if __name__ == "__main__":
    main()
