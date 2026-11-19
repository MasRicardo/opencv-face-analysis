from pathlib import Path

import cv2
import numpy as np

from face_detection import detect_faces, load_detector, make_parser, parse_args, run

AGES = ("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100")
LABELS = ("Male", "Female")
BGR_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


def load_model(folder, kind):
    config = folder / f"{kind}_deploy.prototxt"
    weights = folder / f"{kind}_net.caffemodel"
    if not config.is_file() or not weights.is_file():
        raise SystemExit(f"Model {kind} belum lengkap di {folder}.")
    return cv2.dnn.readNetFromCaffe(str(config), str(weights))


def classify(network, blob, labels):
    network.setInput(blob)
    return labels[int(np.argmax(network.forward()))]


def main():
    parser = make_parser("Estimasi kelompok usia dan klasifikasi label wajah.")
    parser.add_argument("--nama", default="Nama Praktikan", help="Judul jendela.")
    parser.add_argument("--models", type=Path, default=Path(__file__).parent / "models")
    args = parse_args(parser)
    detector = load_detector()
    age_net = load_model(args.models, "age")
    label_net = load_model(args.models, "gender")

    def process(frame):
        faces = detect_faces(frame, detector)
        height, width = frame.shape[:2]
        annotations = []
        for x, y, w, h in faces:
            crop = frame[max(0, y - 20):min(height, y + h + 20),
                         max(0, x - 20):min(width, x + w + 20)]
            blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), BGR_MEAN)
            age = classify(age_net, blob, AGES)
            label = classify(label_net, blob, LABELS)
            annotations.append((x, y, w, h, f"{label}, {age} th"))

        for x, y, w, h, text in annotations:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            left = max(0, min(x, width - size[0] - 8))
            baseline = max(size[1] + 6, y - 8)
            cv2.rectangle(frame, (left, baseline - size[1] - 5),
                          (left + size[0] + 7, baseline + 4), (25, 25, 25), -1)
            cv2.putText(frame, text, (left + 3, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if args.image:
                print(f"Prediksi model: {text}")
        return len(faces)

    run(process, args, args.nama)


if __name__ == "__main__":
    main()
