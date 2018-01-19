#
#   Video Capture
#   Written by Qhan
#

import numpy as np
import cv2
import argparse
import os.path as osp
from hpd import processImage


def main(args):
    filename = args.input_file

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
        out = cv2.VideoWriter(name + '_out.avi', fourcc, fps, (width, height))

    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        print('\rframe: %d' % count, end='')
        ret, frame = cap.read()

        #frame = cv2.resize(frame, (400, 300), cv2.INTER_CUBIC)
        
        if isVideo:
            frame = Process(frame, 1)
            out.write(frame)
        else:
            frame = cv2.flip(frame, 1)
            frame = Process(frame, 1)
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', default=None)
    args = parser.parse_args()
    main(args)
