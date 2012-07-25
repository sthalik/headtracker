#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_initial_guess(headtracker_t& ctx, IplImage& frame, float* rotation_matrix, float* translation_vector) {
	int64 ticks = GetTickCount();

	if (ctx.ticks_last_classification / HT_CLASSIFICATION_DELAY_MS == ticks / HT_CLASSIFICATION_DELAY_MS)
		return false;

	ctx.ticks_last_classification = ticks;

	CvRect rectangles[HT_CLASSIFIER_COUNT];

	if (!ht_classify(ctx, ctx.classifiers[HT_CLASSIFIER_HEAD], frame, cvRect(0, 0, frame.width, frame.height), rectangles[HT_CLASSIFIER_HEAD]))
		return false;

	for (int i = 1; i < HT_CLASSIFIER_COUNT; i++) {
		if (!ht_classify(ctx, ctx.classifiers[i], frame, rectangles[HT_CLASSIFIER_HEAD], rectangles[i]))
			return false;
	}

	CvPoint2D32f image_points[HT_CLASSIFIER_COUNT-1];
	CvPoint3D32f object_points[HT_CLASSIFIER_COUNT-1];

	for (int i = 1; i < HT_CLASSIFIER_COUNT; i++)
		image_points[i-1] = cvPoint2D32f(rectangles[i].x + rectangles[i].width/2, rectangles[i].y + rectangles[i].height/2);

	object_points[HT_CLASSIFIER_NOSE-1] = cvPoint3D32f(0, 0, 0);
	object_points[HT_CLASSIFIER_EYE1-1] = cvPoint3D32f(-30, -13, -39);
	object_points[HT_CLASSIFIER_EYE2-1] = cvPoint3D32f(30, -13, -39);
	object_points[HT_CLASSIFIER_MOUTH-1] = cvPoint3D32f(0, 63, -21);

	return ht_posit(image_points, object_points, HT_CLASSIFIER_COUNT-1, rotation_matrix, translation_vector);
}