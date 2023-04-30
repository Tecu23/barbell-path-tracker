"""
    Steps:
        1. Search reddit for different photos and videos of people doing squats for data modelling.
        2. Detection algorithm.
        3. Tracking algorithm.
"""

import cv2

FILE_PATH = "videos/05ikh6m6jvwa1-DASH_360.mp4" # INPUT VIDEO PATH
OUTPUT_PATH = ""


def main():
    cap = cv2.VideoCapture(FILE_PATH)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
