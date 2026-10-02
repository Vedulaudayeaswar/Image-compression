from io import BytesIO
from pathlib import Path
import os
import subprocess
import sys
from PIL import Image
from backend.pipeline import CompressionConfig, compress, decompress


def test_demo_pipeline_and_independent_dfc_decode():
    image = Image.new("RGB", (32, 24), (80, 120, 180))
    raw = BytesIO(); image.save(raw, format="PNG")
    result = compress(raw.getvalue(), "sample.png", CompressionConfig(mode="demo", resolution=32, steps=1))
    assert result["compression"]["compressed_bytes"] > 0
    assert result["metrics"]["psnr"] > 0
    from backend.config import ARTIFACT_DIR
    decoded, metadata = decompress((ARTIFACT_DIR / f"{result['experiment_id']}.dfc").read_bytes())
    assert decoded.startswith(b"\x89PNG")
    assert metadata["image_size"] == [32, 24]


def test_fresh_process_decodes_dfc_after_source_is_removed(tmp_path: Path):
    source = tmp_path / "original.png"
    Image.new("RGB", (32, 32), (12, 90, 200)).save(source)
    result = compress(source.read_bytes(), source.name, CompressionConfig(mode="demo", resolution=32, steps=1, reconstruction_mode="refinement"))
    from backend.config import ARTIFACT_DIR
    container = tmp_path / "only-input.dfc"
    container.write_bytes((ARTIFACT_DIR / f"{result['experiment_id']}.dfc").read_bytes())
    source.unlink()
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(project))
    program = "from pathlib import Path; from backend.pipeline import decompress; Path('decoded.png').write_bytes(decompress(Path('only-input.dfc').read_bytes())[0])"
    subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=environment, check=True)
    assert (tmp_path / "decoded.png").read_bytes().startswith(b"\x89PNG")
