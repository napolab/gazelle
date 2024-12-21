from typing import Any, Generator, Optional, Self, TypedDict, Unpack
import cv2
import warnings


class CameraDeviceInfo(TypedDict):
    device_id: int
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]


class WebCamera:
    def __init__(
        self,
        **kwargs: Unpack[CameraDeviceInfo],
    ):
        self.capture_device = kwargs['device_id']
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.fps = kwargs['fps']

    def __enter__(self) -> Self:
        print('Opening capture device', self.capture_device)

        self.capture = cv2.VideoCapture(self.capture_device, cv2.CAP_DSHOW)

        if self.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        return self

    def __exit__(self, *args):
        print('Releasing capture device', self.capture_device)

        self.capture.release()

    def stream(self) -> Generator[cv2.typing.MatLike, Any, None]:
        fail_count = 0

        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                fail_count = 0
                yield frame
            else:
                fail_count += 1

            if fail_count >= 5:
                warnings.warn('Failed to read frame')
                break

    def get_frame_width(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_frame_height(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_fps(self):
        return self.capture.get(cv2.CAP_PROP_FPS)
