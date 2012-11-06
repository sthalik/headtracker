#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_initial_guess(headtracker_t& ctx, Mat& frame, Mat& rvec_, Mat& tvec_) {
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

    vector<Point2d> image_points(HT_CLASSIFIER_COUNT-1);
    vector<Point3d> object_points(HT_CLASSIFIER_COUNT-1);

	for (int i = 1; i < HT_CLASSIFIER_COUNT; i++)
        image_points[i-1] = Point2d(rectangles[i].x + rectangles[i].width/2, rectangles[i].y + rectangles[i].height/2);

    object_points[HT_CLASSIFIER_NOSE-1] = Point3d(0, 0, 0);
    object_points[HT_CLASSIFIER_EYE1-1] = Point3d(-15, -29, -31);
    object_points[HT_CLASSIFIER_EYE2-1] = Point3d(15, -29, -31);
    object_points[HT_CLASSIFIER_MOUTH-1] = Point3d(0, 28, -20);

    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = ctx.focal_length;
    intrinsics.at<float> (1, 1) = ctx.focal_length;
    intrinsics.at<float> (0, 2) = ctx.grayscale.cols/2;
    intrinsics.at<float> (1, 2) = ctx.grayscale.rows/2;

    Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);
    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);

    rvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (1, 0) = 1.0;

    solvePnP(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec, false, CV_ITERATIVE);

    rvec_ = rvec;
    tvec_ = tvec;

    return true;
}
