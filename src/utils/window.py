import cv2
from screeninfo import get_monitors


class CVWindow:
    def __init__(self, window_name: str):
        self.name = window_name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, 0, 0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        cv2.destroyWindow(self.name)

    def waitKey(self):
        return cv2.waitKey(1) & 0xFF

    def draw(self, frame: cv2.typing.MatLike) -> None:
        cv2.imshow(self.name, frame)

    def fullscreen(self):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def exit_fullscreen(self):
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def close(self) -> None:
        cv2.destroyWindow(self.name)
