import argparse
from pathlib import Path

import cv2


def load_detector():
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise SystemExit(f"Gagal memuat Haar Cascade: {path}")
    return detector


def detect_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )


def run(process, args, title):
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Gambar tidak dapat dibaca: {args.image}")
        print(f"Jumlah wajah: {process(frame)}")
        if args.output and not cv2.imwrite(str(args.output), frame):
            raise SystemExit(f"Gagal menyimpan gambar: {args.output}")
        if not args.output:
            try:
                cv2.imshow(title, frame)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
        return

    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise SystemExit(f"Kamera {args.camera} tidak dapat dibuka.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Frame kamera gagal dibaca.")
            process(frame)
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, help="Simpan hasil mode gambar.")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.output and not args.image:
        parser.error("--output memerlukan --image")
    return args


def main():
    args = parse_args(make_parser("Deteksi wajah dengan Haar Cascade."))
    detector = load_detector()

    def process(frame):
        faces = detect_faces(frame, detector)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return len(faces)

    run(process, args, "Face Detection")


if __name__ == "__main__":
    main()
