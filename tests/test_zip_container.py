from io import BytesIO
from PIL import Image
from backend.compression.dfc import serialize
from backend.pipeline import unwrap_zip, zip_container


def test_zip_is_only_a_container_and_preserves_dfc():
    dfc = serialize({"version": 1}, b"payload")
    wrapped = zip_container(dfc, "EXP-test")
    assert unwrap_zip(wrapped) == dfc
