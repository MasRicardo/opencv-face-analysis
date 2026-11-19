from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
CONFIG = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/"
WEIGHTS = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/"
SAMPLE = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"


def download(url, path, expected_size):
    if path.is_file() and path.stat().st_size == expected_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Mengunduh {path.name}...")
    try:
        urlretrieve(url, temporary)
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"Ukuran unduhan tidak sesuai: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    models = ROOT / "models"
    for kind, config_bytes, weights_bytes in (
        ("age", 2308, 45661480),
        ("gender", 2309, 45649168),
    ):
        config = f"{kind}_deploy.prototxt"
        weights = f"{kind}_net.caffemodel"
        download(CONFIG + config, models / config, config_bytes)
        download(WEIGHTS + weights, models / weights, weights_bytes)
    download(SAMPLE, ROOT / "sample.jpg", 91814)
    print("Aset siap digunakan.")


if __name__ == "__main__":
    main()
