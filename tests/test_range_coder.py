from backend.entropy.range_coder import decode, encode


def test_arithmetic_round_trip():
    values = [-7, -7, 0, 4, 4, 4, 5, 127, -128] * 8
    assert decode(encode(values)) == values
