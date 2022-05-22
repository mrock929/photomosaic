# Functions for preprocessing the cifar-100 data into  average RGB values for each image
# Data can be found at https://www.cs.toronto.edu/~kriz/cifar.html

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Tuple

NUM_IMAGES = 50000  # there are 50,000 images in Cifar-100 train


def main(input_path: str, output_path: str):
    """
    Main function to load, parse, and determine average RGB values for Cifar-100 image data

    Args:
        input_path: String that contains the path to the input data
        output_path: String that contains the path to save the output data to

    Returns: None

    """
    data_dict = load_data(filename=input_path)
    data_df = convert_data(data_input=data_dict, num_images=NUM_IMAGES)
    save_data(output_path=Path(output_path), output_data=data_df)


def load_data(filename: str) -> dict:
    """
    Load Cifar-100 data from "python format" pickle file

    Args:
        filename: Path to the Cifar-100 data as a string

    Returns: Dict of the extracted Cifar-100 data

    """

    with open(filename, "rb") as fo:
        data = pickle.load(fo, encoding="latin1")

    return data


def convert_data(data_input: dict, num_images: int) -> pd.DataFrame:
    """
    Convert the data from the raw Cifar-100 format to a dataframe with columns of id, average RGB, coarse class, and
        fine class

    Args:
        data_input: Input data
        num_images: Number of images to convert

    Returns: Cleaned and converted dataframe with relevant information

    """

    data = initialize_dataframe(input_dict=data_input)

    # Preallocated arrays
    avg_red = np.zeros(
        num_images
    )  # There are 3 color channels: red, green, and blue (RGB)
    avg_green = np.zeros(num_images)
    avg_blue = np.zeros(num_images)

    # Convert RGB array stored in data_input["data"] into average RGB for each image
    for i, array_data in enumerate(data_input["data"]):
        r, g, b = get_avg_rgb(rgb_array=array_data)
        avg_red[i] = r
        avg_green[i] = g
        avg_blue[i] = b

    data["avg_red"] = avg_red  # average red values. Range from 0-255
    data["avg_green"] = avg_green  # average green values. Range from 0-255
    data["avg_blue"] = avg_blue  # average blue values. Range from 0-255

    return data


def initialize_dataframe(input_dict: dict) -> pd.DataFrame:
    """
    Initialize the dataframe to contain all relevant info and average RGB values

    Args:
        input_dict: Cifar-100 dictionary loaded from file

    Returns: Dataframe extracted from Cifar-100 dictionary

    """

    data = pd.DataFrame()
    data["filename"] = input_dict["filenames"]
    data["coarse_labels"] = input_dict["coarse_labels"]
    data["fine_labels"] = input_dict["fine_labels"]

    return data


def get_avg_rgb(rgb_array: np.array) -> Tuple[int, int, int]:
    """
    Convert the Cifar-100 data into average RGB for a given image

    Args:
        rgb_array: Input RGB array that contains the RGB values for each pixel in an image (32x32 pixels)

    Returns: Tuple with the average Red, Green, and Blue values for the image

    """

    average_red = round(np.average(rgb_array[0:1024]))
    average_green = round(np.average(rgb_array[1024:2048]))
    average_blue = round(np.average(rgb_array[2048:]))

    return average_red, average_green, average_blue


def save_data(output_path: Path, output_data: pd.DataFrame) -> None:
    """
    Save the dataframe to file

    Args:
        output_path: Path to save the file to
        output_data: The data to save to file

    Returns: None

    """

    output_data.to_csv(output_path)


if __name__ == "__main__":
    main(
        input_path="./cifar-100-python/train",
        output_path="./avg_rgb.csv",
    )
