import numpy as np

from pipeline.features import get_feature_match_ratio
from pipeline.features import get_laplacian_variance
from pipeline.features import is_well_exposed


def test_get_laplacian_variance():
    """
    Test the get_laplacian_variance function.

    This function verifies that the Laplacian variance calculation
    returns a float and that it correctly processes a dummy image.
    """
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    laplacian_var = get_laplacian_variance(image)

    assert isinstance(
        laplacian_var, float
    ), "Laplacian variance should be a float"
    assert laplacian_var >= 0, "Laplacian variance should be non-negative"


def test_is_well_exposed():
    """
    Test the is_well_exposed function.

    This function verifies that the well-exposed check returns a boolean
    and correctly processes a dummy image.
    """
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    well_exposed = is_well_exposed(image)
    print(type(well_exposed))

    assert isinstance(
        well_exposed, (bool, np.bool_)
    ), "Well-exposed check should return a boolean"


def test_get_feature_match_ratio():
    """
    Test the get_feature_match_ratio function.

    This function verifies that the feature match ratio calculation
    returns a float and correctly processes two dummy images.
    """
    image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    match_ratio = get_feature_match_ratio(image1, image2)

    assert isinstance(
        match_ratio, float
    ), "Feature match ratio should be a float"
    assert (
        0 <= match_ratio <= 1
    ), "Feature match ratio should be between 0 and 1"

    # Additional value check
    if match_ratio == 0:
        print("Warning: No features matched between the images.")
