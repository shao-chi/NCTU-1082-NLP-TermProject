import cv2

def gif_save_image(gif_path, save_path, image_size):
    video_capture = cv2.VideoCapture(gif_path)

    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    # size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    _, frame = video_capture.read()
    frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, frame)

    video_capture.release()

    return frame