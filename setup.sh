wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -vd shape_predictor_68_face_landmarks.dat.bz2
mkdir -vp model
mv -v shape_predictor_68_face_landmarks.dat model
