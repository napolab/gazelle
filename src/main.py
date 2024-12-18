from typing import Dict, List
import cv2
import torch
from utils.web_camera import CameraDeviceInfo, WebCamera
from utils.window import CVWindow
from retinaface.pre_trained_models import get_model

device_info = CameraDeviceInfo(device_id=0, width=1920, height=1080, fps=60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model('resnet50_2020-07-20', max_size=2048, device=str(device))
model.eval()


def draw_faces(frame: cv2.typing.MatLike, detections: List[Dict[str, List | float]]):
    """検出結果を描画"""
    for detection in detections:
        bbox = detection['bbox']
        landmarks = detection['landmarks']
        score = detection['score']

        if score < 0:  # 無効なスコアはスキップ
            continue

        x_min, y_min, x_max, y_max = list(map(lambda x: int(x), bbox))
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'{score:.2f}',
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        for x, y in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)


def main():
    with WebCamera(**device_info) as camera, CVWindow('web_camera') as window:
        for frame in camera.stream():
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('f'):
                window.fullscreen()
            if key == 27:
                window.exit_fullscreen()

            detections = model.predict_jsons(
                frame, confidence_threshold=0.7, nms_threshold=0.4
            )
            draw_faces(frame, detections)

            window.draw(frame)


if __name__ == '__main__':
    main()
