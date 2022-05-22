# Tests for preprocessing.py

from .preprocessing import convert_data, initialize_dataframe, get_avg_rgb
import numpy as np


def test_convert_data():
    # Arrange
    rgb_array = np.ones(3072)
    input_dict = {"filenames": ["a", "b"], "coarse_labels": ["c", "d"], "fine_labels": ["e", "f"], "data": [rgb_array, 5 * rgb_array]}

    # Act
    output = convert_data(data_input=input_dict, num_images=2)

    # Assert
    assert np.all(output["filename"] == input_dict["filenames"])
    assert np.all(output["coarse_labels"] == input_dict["coarse_labels"])
    assert np.all(output["fine_labels"] == input_dict["fine_labels"])
    assert np.allclose(output["avg_red"], [1.0, 5.0], atol=1e-6)
    assert np.allclose(output["avg_green"], [1.0, 5.0], atol=1e-6)
    assert np.allclose(output["avg_blue"], [1.0, 5.0], atol=1e-6)


def test_initialize_dataframe():
    # Arrange
    input_dict = {"filenames": ["a", "b"], "coarse_labels": ["c", "d"], "fine_labels": ["e", "f"]}

    # Act
    output = initialize_dataframe(input_dict=input_dict)

    # Assert
    assert np.all(output["filename"] == input_dict["filenames"])
    assert np.all(output["coarse_labels"] == input_dict["coarse_labels"])
    assert np.all(output["fine_labels"] == input_dict["fine_labels"])
    assert "data" not in output.columns


def test_get_avg_rgb():
    # Arrange
    r_array = np.ones(1024)
    g_array = 2 * np.ones(1024)
    b_array = 3 * np.ones(1024)
    rgb_array = np.concatenate((r_array, g_array, b_array))

    # Act
    avg_r, avg_g, avg_b = get_avg_rgb(rgb_array=rgb_array)

    # Assert
    assert avg_r == 1
    assert avg_g == 2
    assert avg_b == 3
