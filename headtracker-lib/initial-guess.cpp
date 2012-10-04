#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_initial_guess(headtracker_t& ctx, Mat& frame, float* rotation_matrix, float* translation_vector) {
	int ticks = ht_tickcount();

	if (ctx.ticks_last_classification / ctx.config.classification_delay == ticks / ctx.config.classification_delay)
		return false;

	ctx.ticks_last_classification = ticks;

    Rect rectangles[HT_CLASSIFIER_COUNT];

    if (!ht_classify(ctx.classifiers[HT_CLASSIFIER_HEAD], frame, Rect(0, 0, frame.cols, frame.rows), rectangles[HT_CLASSIFIER_HEAD]))
		return false;

	for (int i = 1; i < HT_CLASSIFIER_COUNT; i++)
		if (!ht_classify(ctx.classifiers[i], frame, rectangles[HT_CLASSIFIER_HEAD], rectangles[i]))
			return false;

	CvPoint2D32f image_points[HT_CLASSIFIER_COUNT-1];
	CvPoint3D32f object_points[HT_CLASSIFIER_COUNT-1];

	for (int i = 1; i < HT_CLASSIFIER_COUNT; i++)
		image_points[i-1] = cvPoint2D32f(rectangles[i].x + rectangles[i].width/2, rectangles[i].y + rectangles[i].height/2);

	object_points[HT_CLASSIFIER_NOSE-1] = cvPoint3D32f(0, 0, 0);
    object_points[HT_CLASSIFIER_EYE1-1] = cvPoint3D32f(-23.23, -29, -33.5);
    object_points[HT_CLASSIFIER_EYE2-1] = cvPoint3D32f(23.23, -29, -33.5);
    object_points[HT_CLASSIFIER_MOUTH-1] = cvPoint3D32f(0, 29, -22.31);

	return ht_posit(image_points,
					object_points,
					HT_CLASSIFIER_COUNT-1,
					rotation_matrix,
					translation_vector,
                    cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-8),
					ctx.focal_length);
}
