from typing import Dict, List
import cv2
import numpy as np
import torch
from utils.web_camera import CameraDeviceInfo, WebCamera
from utils.window import CVWindow
from retinaface.pre_trained_models import get_model
from hubconf import gazelle_dinov2_vitl14_inout
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

device_info = CameraDeviceInfo(device_id=0, width=1920, height=1080, fps=60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
retina_face_model = get_model('resnet50_2020-07-20', max_size=2048, device=str(device))
retina_face_model.eval()

model, transform = gazelle_dinov2_vitl14_inout()
model.eval().to(device)


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


def visualize_all(cv_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    colors = [
        (50, 205, 50),
        (255, 99, 71),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]  # BGR
    overlay_image = cv_image
    height, width, _ = cv_image.shape

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        thickness = int(min(width, height) * 0.01)

        # Draw rectangle (bounding box)
        cv2.rectangle(
            overlay_image,
            (int(xmin * width), int(ymin * height)),
            (int(xmax * width), int(ymax * height)),
            color,
            thickness,
        )

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f'in-frame: {inout_score:.2f}'
            text_x = int(xmin * width)
            text_y = int(ymax * height + thickness * 2)
            font_scale = min(width, height) * 0.002
            font_thickness = max(1, int(thickness * 0.5))

            # Put text (in/out score)
            cv2.putText(
                overlay_image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            # Draw gaze target (circle)
            gaze_radius = int(0.005 * min(width, height))
            cv2.circle(
                overlay_image,
                (int(gaze_target_x), int(gaze_target_y)),
                gaze_radius,
                color,
                -1,
            )

            # Draw line from bounding box center to gaze target
            cv2.line(
                overlay_image,
                (int(bbox_center_x), int(bbox_center_y)),
                (int(gaze_target_x), int(gaze_target_y)),
                color,
                thickness=max(1, gaze_radius),
            )

    return overlay_image


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

            face_detections = retina_face_model.predict_jsons(
                frame, confidence_threshold=0.7, nms_threshold=0.4
            )
            draw_faces(frame, face_detections)

            bboxes = [detection['bbox'] for detection in face_detections]
            img_tensor = transform(frame).unsqueeze(0).to(device)

            width = camera.get_frame_width()
            height = camera.get_frame_height()
            norm_bboxes = [
                np.array(bbox) / np.array([width, height, width, height])
                for bbox in bboxes
                if len(bbox) == 4
            ]
            if len(norm_bboxes) > 0:
                input = {'images': img_tensor, 'bboxes': [norm_bboxes]}
                with torch.no_grad():
                    output = model(input)
                visualize_all(
                    frame,
                    output['heatmap'][0],
                    norm_bboxes,
                    output['inout'][0] if output['inout'] is not None else None,
                    inout_thresh=0.5,
                )

            window.draw(frame)


if __name__ == '__main__':
    main()
