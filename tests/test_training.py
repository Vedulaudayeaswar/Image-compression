from pathlib import Path
from PIL import Image
from backend.training.runner import LocalImageDataset


def test_local_image_dataset_samples_normalized_square_crop(tmp_path: Path):
    Image.new("RGB", (16, 8), (20, 40, 60)).save(tmp_path / "sample.png")
    dataset = LocalImageDataset(str(tmp_path), crop_size=12)
    import random
    tensor = dataset.sample(random.Random(1))
    assert tensor.shape == (3, 12, 12)
    assert dataset.summary()["image_count"] == 1
