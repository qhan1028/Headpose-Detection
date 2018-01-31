#
#   Video Capture
#   Written by Qhan
#

import numpy as np
import cv2
import argparse
import os.path as osp
from hpd import HPD


def main(args):
    filename = args["input_file"]

    if filename is None:
        isVideo = False
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
    else:
        isVideo = True
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Initialize head pose detection
    hpd = HPD(args["landmark_type"], args["landmark_predictor"])

    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        print('\rframe: %d' % count, end='')
        ret, frame = cap.read()
        
        if isVideo:
            frame, angles = hpd.processImage(frame)
            if frame is None: 
                break
            else:
                out.write(frame)
        else:
            frame = cv2.flip(frame, 1)
            frame, angles = hpd.processImage(frame)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1

    # When everything done, release the capture
    cap.release()
    if isVideo: out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
