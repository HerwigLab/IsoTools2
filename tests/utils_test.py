from isotools._utils import has_overlap, get_overlap


def test_has_overlap_flipped_coordinates():
    # regression test for #23: coordinates on reverse-strand features are
    # sometimes passed as (end, start) rather than (start, end) -- both
    # orderings must give the same result
    assert has_overlap((90, 118), (100, 200)) is True
    assert has_overlap((118, 90), (100, 200)) is True  # previously wrongly False
    assert has_overlap((10, 20), (100, 200)) is False
    assert has_overlap((20, 10), (100, 200)) is False


def test_get_overlap_flipped_coordinates():
    assert get_overlap((90, 118), (100, 200)) == 18
    assert get_overlap((118, 90), (100, 200)) == 18  # previously wrong result
    assert get_overlap((10, 20), (100, 200)) == 0
    assert get_overlap((20, 10), (100, 200)) == 0
