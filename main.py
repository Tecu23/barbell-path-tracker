"""
    Steps:
        1. Search reddit for different photos and videos of people doing squats for data modelling.
        2. Detection algorithm.
        3. Tracking algorithm.
"""

import cv2

FILE_PATH = "videos/05ikh6m6jvwa1-DASH_360.mp4"  # INPUT VIDEO PATH
OUTPUT_PATH = "output/test.mp4"


def createVideoWriter(video):
    # get size of the original video in otder to make output video same size
    sizeOfOutput = (int(video.get(3)), int(video.get(4)))
    outputName = OUTPUT_PATH
    output = cv2.VideoWriter(
        outputName, cv2.VideoWriter_fourcc(*"mp4v"), 30, sizeOfOutput
    )

    return output


def main():
    cap = cv2.VideoCapture(FILE_PATH)

    tracker = cv2.legacy.TrackerKCF_create()

    videoWriter = createVideoWriter(cap)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    ret, frame = cap.read()

    cv2.imshow("Frame", frame)

    boundingBox = cv2.selectROI("Frame", frame)

    try:
        tracker.init(frame, boundingBox)
    except:
        print("A bounding box was not correctly created. Please try again.")

    centerPoints = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        (success, box) = tracker.update(frame)

        if success:
            # gives coords for bounding box
            (x, y, w, h) = [int(i) for i in box]
            # draw it
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # gives the pos of the center of the object to later show its prev positions
            centerOfRectangle = (int(x + w / 2), int(y + h / 2))
            # this circle drawn in the center shows the center
            cv2.circle(
                img=frame,
                center=centerOfRectangle,
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )
            # we append the the center of the object to later then be able to draw prev positions
            centerPoints.append(centerOfRectangle)
            # go thru the list of center points
            for i in range(len(centerPoints)):
                # we are drawing lines through the first point recorded to the most recent
                # this if statement makes sure that we dont connect the last point to the first creating a polygon
                if (i - 1) == -1:
                    continue
                # connect points with a line
                cv2.line(
                    frame,
                    pt1=centerPoints[i - 1],
                    pt2=centerPoints[i],
                    color=(0, 0, 255),
                    thickness=2,
                )
            # write this frame to output video
            videoWriter.write(frame)
            # shwo the frame
            cv2.imshow("Frame", frame)

            # this code allows us to assign a key that would break the object tracking
            # in this use case it is not needed but it is nice to have
            # KEY IS "q"
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
