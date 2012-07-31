#include "stdafx.h"

using namespace std;
using namespace cv;

classifier_t ht_make_classifier(const char* filename, rect_t rect, CvSize2D32f min_size)
{
	classifier_t ret;

	ret.cascade = (CvHaarClassifierCascade*) cvLoad(filename);
	if (ret.cascade == NULL)
		throw exception();
	ret.rect = rect;
	ret.min_size = min_size;

	return ret;
}

bool ht_classify(const classifier_t& classifier, IplImage& frame, const CvRect& roi, CvRect& ret) {
	CvRect roi2 = roi;

	roi2.x += roi.width * classifier.rect.x;
	roi2.y += roi.height * classifier.rect.y;
	roi2.width *= classifier.rect.w;
	roi2.height *= classifier.rect.h;

	if (!(roi2.width > 0 && roi2.height > 0 && roi2.x >= 0 && roi2.y >= 0 && roi2.x + roi2.width <= frame.width && roi2.y + roi2.height <= frame.height))
		return false;

	cvSetImageROI(&frame, roi2);
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* seq = cvHaarDetectObjects(&frame,
									 classifier.cascade,
									 storage,
									 1.18,
									 2,
									 CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT,
									 cvSize(1 + classifier.min_size.width * roi2.width, 1 + classifier.min_size.height * roi2.height),
									 cvSize(roi2.width, roi2.height));
	int size = -1;

	for (int i = 0; i < seq->total; i++) {
		CvRect rect = *(CvRect*)cvGetSeqElem(seq, i);
		if (size < rect.width * rect.height) {
			size = rect.width * rect.height;
			rect.x += roi2.x;
			rect.y += roi2.y;
			ret = rect;
		}
	}
	cvResetImageROI(&frame);
	cvReleaseMemStorage(&storage);
	return size > 0;
}

void ht_free_classifier(classifier_t* classifier) {
	cvReleaseHaarClassifierCascade(&classifier->cascade);
}