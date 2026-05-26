from src.utils.rareness import compute_rareness_factors


def test_rare_case_gets_higher_factor():
    # Rare code Z appears once; common code B appears in every doc
    labels = [["Z", "B"], ["B"], ["B"]]
    factors = compute_rareness_factors(labels)
    assert factors[0] > factors[1]
