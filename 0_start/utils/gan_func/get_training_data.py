from PIL import Image
import os
from tqdm import tqdm
import numpy as np


def get_training_data(datafolder, image_width, image_height, channels):
    """
    Loads the training data from a specified folder

    Parameters
    ----------
    datafolder : String
        Path to directory where the training images are stored
    image_width : Integer
        Pixel width of the images (Nr. of columns)
    image_height : Integer
        Pixel height of the images (Nr. of rows)
    channels : Integer
        Number of color channels

    Returns
    -------
    training_data : 4-dim numpy array
        The training data as numpy array of shaape:
            (Nr. images, image_height, image_width, Nr. channels)

    """
    print("Loading training data...")

    training_data = []
    # Finds all files in datafolder
    filenames = os.listdir(datafolder)
    filenames.remove('.gitkeep')
    for filename in tqdm(filenames):
        # Combines folder name and file name.
        path = os.path.join(datafolder, filename)
        # Opens an image as an Image object.
        image = Image.open(path)
        # Resizes to a desired size.
        image = image.resize((image_width, image_height), Image.ANTIALIAS)
        # Creates an array of pixel values from the image.
        pixel_array = np.asarray(image)

        # Clip alpha channel, if existant
        if pixel_array.shape[2] > channels:
            pixel_array = pixel_array[:, :, :channels]

        training_data.append(pixel_array)

    # training_data is converted to a numpy array
    training_data = \
        np.reshape(training_data, (-1, image_width, image_height, channels))
    return training_data
