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
    else:
        isVideo = True
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))

    count = 0
    hpd = HPD(args["landmark_type"], args["landmark_predictor"], args["box_length"])
    while(cap.isOpened()):
        # Capture frame-by-frame
        print('\rframe: %d' % count, end='')
        ret, frame = cap.read()

        #frame = cv2.resize(frame, (400, 300), cv2.INTER_CUBIC)
        
        if isVideo:
            frame = hpd.processImage(frame)
            if frame is None: 
                break
            else:
                out.write(frame)
        else:
            frame = cv2.flip(frame, 1)
            frame = hpd.processImage(frame)

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
    parser.add_argument('-i', dest='input_file', default=None)
    parser.add_argument('-o', dest='output_file', default=None)
    parser.add_argument('-lt', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', dest='landmark_predictor', default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    parser.add_argument('-b', dest='box_length', type=float, default=10.0)
    args = vars(parser.parse_args())
    main(args)
