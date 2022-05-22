# Test the create_mosaic.py file

from pytest import mark
from pytest_mock import MockFixture
from create_mosaic import CreateMosaic
from PIL import Image
import numpy as np
import pandas as pd


def test_init():
    # Arrange / Act
    mosaic = CreateMosaic()

    # Assert
    assert isinstance(mosaic.original_image, type(Image))
    assert mosaic.width == 0
    assert mosaic.height == 0


@mark.parametrize("coarse_class, calls", [("trees", 1), ("all", 0)])
def test_create_mosaic(coarse_class: str, calls: int, mocker: MockFixture):
    # Arrange
    mock_load = mocker.patch.object(CreateMosaic, "_load_image", return_value=Image.open("./data/mario.jpg"))
    mock_load_avg = mocker.patch.object(CreateMosaic, "_load_avg_cifar_data")
    mock_trim = mocker.patch.object(CreateMosaic, "_trim_cifar_data_by_class")
    mock_load_full = mocker.patch.object(CreateMosaic, "_load_full_cifar_data")
    mock_generate = mocker.patch.object(CreateMosaic, "_generate_mosaic")
    mock_save = mocker.patch.object(CreateMosaic, "_save_image")

    # Act
    mosaic = CreateMosaic()
    mosaic.create_mosaic(image_path="./", output_path="./", coarse_class=coarse_class)

    # Assert
    assert mock_load.call_count == 1
    assert mock_load_avg.call_count == 1
    assert mock_trim.call_count == calls
    assert mock_load_full.call_count == 1
    assert mock_generate.call_count == 1
    assert mock_save.call_count == 1
    assert mosaic.width == 256
    assert mosaic.height == 256


def test_load_image():
    # Arrange
    input_path = "./data/mario.jpg"
    mosaic = CreateMosaic()

    # Act
    output = mosaic._load_image(input_path=input_path)
    width, height = output.size

    # Assert
    assert width == 256
    assert height == 256


def test_load_avg_cifar_data():
    # Arrange / Act
    mosaic = CreateMosaic()
    output = mosaic._load_avg_cifar_data()

    # Assert
    assert len(output["avg_red"]) == 50000


def test_trim_cifar_data_by_class():
    # Arrange
    cifar_data = pd.DataFrame({"coarse_labels": [1, 2, 1]})

    # Act
    mosaic = CreateMosaic()
    output = mosaic._trim_cifar_data_by_class(cifar_data=cifar_data, trim_class="fish")

    # Assert
    assert len(output["coarse_labels"]) == 2


def test_load_full_cifar_data():
    # Arrange / Act
    mosaic = CreateMosaic()
    output = mosaic._load_full_cifar_data()

    # Assert
    assert len(output["filenames"]) == 50000
    assert len(output["data"]) == 50000


def test_generate_mosaic(mocker: MockFixture):
    # Arrange
    mosaic = CreateMosaic()
    mosaic.width = 3
    mosaic.height = 5

    original_image = np.zeros((mosaic.height, mosaic.width, 3))
    out_image = np.zeros((mosaic.height * 32, mosaic.width * 32, 3))
    fake_image = np.ones((32, 32, 3))

    mock_initialize = mocker.patch.object(mosaic, "_initialize_arrays", return_value=(original_image, out_image))
    mock_find = mocker.patch.object(mosaic, "_find_best_image", return_value=fake_image)

    # Act
    output = mosaic._generate_mosaic(avg_im_data=pd.DataFrame(), full_im_data=pd.DataFrame())
    output_array = np.asarray(output)

    # Assert
    assert np.all(output_array == 1)
    assert mock_initialize.call_count == 1
    assert mock_find.call_count == mosaic.width * mosaic.height


def test_initialize_arrays():
    # Arrange
    mosaic = CreateMosaic()
    mosaic.width = 3
    mosaic.height = 5

    original_image = 2.0 * np.ones((mosaic.height, mosaic.width, 3))
    mosaic.original_image = Image.fromarray(original_image.astype(np.uint8), mode="RGB")

    # Act
    original_array, out_array = mosaic._initialize_arrays()
    out_size = out_array.shape

    # Assert
    assert np.allclose(original_image, original_array, atol=1e-6)
    assert out_size[0] == mosaic.height * 32
    assert out_size[1] == mosaic.width * 32
    assert np.sum(out_array) == 0
