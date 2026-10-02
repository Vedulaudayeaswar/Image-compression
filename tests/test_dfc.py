import pytest
from backend.compression.dfc import deserialize, serialize


def test_dfc_round_trip_and_checksum():
    blob = serialize({"version": 1, "latent_shape": [1, 4, 2, 2]}, b"compressed")
    metadata, payload = deserialize(blob)
    assert metadata["version"] == 1 and payload == b"compressed"
    with pytest.raises(ValueError): deserialize(blob[:-1] + b"x")
