import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir, path


def resize_image(image: Image):
    """
    Resize the image
    :param image: Input, will not be mutated
    :return: Resized image
    """
    old_width, old_height = image.size

    square_size = min(old_width, old_height)
    output_size = 128

    left = (old_width - square_size) / 2
    top = (old_height - square_size) / 2
    right = (old_width + square_size) / 2
    bottom = (old_height + square_size) / 2

    image = image.crop((left, top, right, bottom))
    image.thumbnail((output_size, output_size), Image.ANTIALIAS)
    return image


def read_images(folder: str, force_resize: bool = False):
    """
    Read image dataset from folder. The label will be the image's name up to the first underscore
    :param folder:
    :param force_resize: ignore input files with .thumb.jpg extension
    :return: images, labels
    """
    names = listdir(folder)

    images, labels = [], []

    for name in names:
        if name.endswith(".thumb.jpg"):
            if force_resize:
                continue  # Don't use premade thumb
            image = Image.open(path.join(folder, name))
        else:
            thumb_name = path.splitext(name)[0] + ".thumb.jpg"
            if not force_resize and thumb_name in names:
                continue  # Resized version exists, use that instead
            image = Image.open(path.join(folder, name))
            image = resize_image(image)
            image.save(path.join(folder, thumb_name))  # Save a resized version

        label = name.split('_')[0]
        labels.append(label)
        images.append(image)

    return images, labels


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def numpy_grayscale(X):
    return np.array([rgb2gray(X[i]) for i in range(X.shape[0])])


def plot_gallery(images, titles, h=128, w=128, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if titles is not None:
            plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()