#include "stdafx.h"

using namespace std;
using namespace cv;

classifier_t ht_make_classifier(const char* filename, rect_t rect, CvSize2D32f min_size)
{
	classifier_t ret;

    ret.cascade.load(filename);
	ret.rect = rect;
	ret.min_size = min_size;

	return ret;
}

bool ht_classify(classifier_t& classifier, Mat& frame, const Rect& roi, Rect& ret) {
    Rect roi2 = roi;

	roi2.x += roi.width * classifier.rect.x;
	roi2.y += roi.height * classifier.rect.y;
	roi2.width *= classifier.rect.w;
	roi2.height *= classifier.rect.h;

    if (!(roi2.width > 0 && roi2.height > 0 && roi2.x >= 0 && roi2.y >= 0 && roi2.x + roi2.width <= frame.cols && roi2.y + roi2.height <= frame.rows))
		return false;

    Mat frame2 = frame(roi2);
    vector<Rect> seq;
    classifier.cascade.detectMultiScale(frame2,
                                        seq,
                                        1.15,
                                        1,
                                        CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT);

    if (seq.size() > 0) {
        seq[0].x += roi2.x;
        seq[0].y += roi2.y;
        ret = seq[0];
    }
    return seq.size() > 0;
}
