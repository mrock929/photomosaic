from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import pickle
from tqdm import tqdm


class CreateMosaic:
    """Class to create a photomosaic from an input image using the Cifar-100 dataset:
    https://www.cs.toronto.edu/~kriz/cifar.html"""

    def __init__(self):
        """Initialize the class and class variables"""

        self.original_image, self.width, self.height = Image, 0, 0

    def create_mosaic(
        self, image_path: str, output_path: str, coarse_class: Optional[str] = "all"
    ) -> None:
        """
        Initialize the class and load the image to memory

        Args:
            image_path: String path to the image to make a mosaic of
            output_path: String path to the location and filename to save the mosaic to
            coarse_class: String containing a superclass defined on the Cifar website to filter by
        """

        # Load input image
        self.original_image = self._load_image(input_path=image_path)

        # Store image width and height
        self.width, self.height = self.original_image.size

        # Load average RGB data
        avg_cifar_data = self._load_avg_cifar_data()

        # Trim the average RGB data to only include class of interest
        if not coarse_class == "all":
            avg_cifar_data = self._trim_cifar_data_by_class(
                cifar_data=avg_cifar_data, trim_class=coarse_class
            )

        # Load the full Cifar-100 dataset
        full_cifar_data = self._load_full_cifar_data()

        # Generate the mosaic
        output_image = self._generate_mosaic(
            avg_im_data=avg_cifar_data, full_im_data=full_cifar_data
        )

        # Save the mosaic to file
        self._save_image(out_path=output_path, out_image=output_image)

    @staticmethod
    def _load_image(input_path: str) -> Image:
        """
        Load input image from a string path

        Args:
            input_path: String representation of the input path

        Returns: PIL Image loaded from file

        """

        image = Image.open(input_path)

        # Convert to RGB
        image = image.convert(mode="RGB", colors=256)

        return image

    @staticmethod
    def _load_avg_cifar_data() -> pd.DataFrame:
        """
        Load Cifar-100 average RGB information from file

        Returns: Dataframe of the Cifar-100 average RGB data generated in ./data/preprocessing.py

        """

        return pd.read_csv("./data/avg_rgb.csv")

    @staticmethod
    def _trim_cifar_data_by_class(
        cifar_data: pd.DataFrame, trim_class: str
    ) -> pd.DataFrame:
        """
        Filter the Cifar-100 data by coarse class. Only return the portions of the DataFrame with the right class

        Args:
            cifar_data: Avg RGB values and classes for the Cifar-100 data.

        Returns: Trimmed average RGB Cifar-100 data in a dataframe

        """

        class_mapping = {
            "aquatic mammals": 0,
            "fish": 1,
            "flowers": 2,
            "food containers": 3,
            "fruit and vegetables": 4,
            "household electrical devices": 5,
            "household furniture": 6,
            "insects": 7,
            "large carnivores": 8,
            "large man-made outdoor things": 9,
            "large natural outdoor scenes": 10,
            "large omnivores and herbivores": 11,
            "medium-sized mammals": 12,
            "non-insect invertebrates": 13,
            "people": 14,
            "reptiles": 15,
            "small mammals": 16,
            "trees": 17,
            "vehicles 1": 18,
            "vehicles 2": 19,
        }

        return cifar_data.loc[cifar_data["coarse_labels"] == class_mapping[trim_class]]

    @staticmethod
    def _load_full_cifar_data() -> Dict[str, np.array]:
        """
        Load Cifar-100 average RGB information from file

        Returns: Dictionary of the full Cifar-100 RGB data

        """

        filename = "./data/cifar-100-python/train"

        with open(filename, "rb") as fo:
            data = pickle.load(fo, encoding="latin1")

        return {"filenames": data["filenames"], "data": data["data"]}

    def _generate_mosaic(
        self, avg_im_data: pd.DataFrame, full_im_data: np.array
    ) -> Image:
        """
        Generate a mosaic from the original image where each pixel in the original has been replaced by the appropriate
        Cifar-100 image

        Args:
            avg_im_data: The available 32x32 pixel Cifar-100 images for creating the mosaic from
            full_im_data: The filenames and corresponding RGB data for Cifar-100 images

        Returns: PIL Image where each pixel of the original image is a Cifar-100 image

        """

        # Generate the numpy arrays for original and output images
        im_array, out_im_array = self._initialize_arrays()

        # For each pixel, find the image with the closest average RGB values and place it in the correct place in out_im
        for i in tqdm(
            range(int(self.height))
        ):  # This is row index (y axis, top to bottom)
            for j in range(
                int(self.width)
            ):  # This is column index (x axis, left to right)

                # Get the RGB values for the pixel of interest
                pixel_rgb = im_array[i, j, :]

                best_image = self._find_best_image(rgb=pixel_rgb, avg_data=avg_im_data, full_data=full_im_data)

                # Insert array into correct location in output image RGB array
                out_im_array[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32, :] = best_image

        return Image.fromarray(out_im_array.astype(np.uint8), mode="RGB")

    def _initialize_arrays(self) -> Tuple[np.array, np.array]:
        """
        Initialize the numpy arrays of the input and output images

        Returns: Tuple with the input image numpy array and the empty output image numpy array

        """

        # for pixel in image, determine closest cifar-100 image, insert into final.
        im_array = np.asarray(self.original_image)

        # Create new output image which has the resolution of the original image times 32x32
        out_im = np.zeros((self.height * 32, self.width * 32, 3))

        return im_array, out_im

    @staticmethod
    def _find_best_image(rgb: np.array, avg_data: pd.DataFrame, full_data: dict) -> np.array:
        """
        Find the best Cifar-100 image based on the input pixel RGB values. The best image has the closest average RGB
        values to the pixel RGB values in a sum of the squares sense

        Args:
            rgb: RGB values of the pixel in the original image
            avg_data: Average RGB data for Cifar-100
            full_data: Full RGB data for Cifar-100

        Returns: Array containing the relevant RGB information for the best image

        """

        # Find minimum sum of the squares between average RGB channels in im_data and pixel value
        # Calculate all squares for each color
        r_diff = np.square(avg_data["avg_red"] - rgb[0])
        g_diff = np.square(avg_data["avg_green"] - rgb[1])
        b_diff = np.square(avg_data["avg_blue"] - rgb[2])

        # Create RGB diff (squared) array, find the sum of each row (sum of RGB for each image), and find the
        # image with the lowest sum of the squares
        best_index = np.argmin(
            np.sum(np.stack((r_diff, g_diff, b_diff), axis=0), axis=0)
        )

        # Grab the corresponding image name
        best_file = avg_data.iloc[best_index]["filename"]

        # Find index of best file
        full_index = full_data["filenames"].index(best_file)

        # Grab the RGB info for this file from original data
        rgb_data = full_data["data"][full_index]

        # Construct the 32x32x3 numpy array of RGB values
        out_array = np.stack(
            (rgb_data[0:1024], rgb_data[1024:2048], rgb_data[2048:]), axis=1
        ).reshape(32, 32, 3)

        return out_array

    @staticmethod
    def _save_image(out_path: str, out_image: Image) -> None:
        """
        Save the mosaic image to file

        Args:
            out_path: Path to save the image to. This should include the file extension
            out_image: PIL Image to save to file

        """

        out_image.show()

        out_image.save(out_path)


if __name__ == "__main__":
    mosaic = CreateMosaic()
    mosaic.create_mosaic(
        image_path="/Users/mrockwood/Projects/photomosaic/data/mario.jpg",
        output_path="/Users/mrockwood/Projects/photomosaic/data/mario_mosaic_fish.png",
        coarse_class="fish",
    )
