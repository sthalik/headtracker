#include "ht-api.h"
#include "ht-internal.h"
using namespace std;
using namespace cv;

bool ht_classify(CascadeClassifier &classifier, Mat& frame, Rect& ret) {
    vector<Rect> seq;
    classifier.detectMultiScale(frame,
                                seq,
                                1.1,
                                3,
                                CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT,
                                Size(80, 80));

    if (seq.size() > 0)
        ret = seq[0];

    return seq.size() > 0;
}
