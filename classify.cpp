#include "stdafx.h"

using namespace std;
using namespace cv;

classifier_t ht_make_classifier(const char* filename, rect_t rect)
{
	classifier_t ret;

    ret.cascade.load(filename);
	ret.rect = rect;

	return ret;
}

bool ht_classify(classifier_t& classifier, Mat& frame, Rect& ret) {
    vector<Rect> seq;
    classifier.cascade.detectMultiScale(frame,
                                        seq,
                                        1.1,
                                        2,
                                        CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH);

    if (seq.size() > 0)
        ret = seq[0];

    return seq.size() > 0;
}
