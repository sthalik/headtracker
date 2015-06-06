#include "ht-api.h"
#include "ht-internal.h"
#include <algorithm>
using namespace std;
using namespace cv;

#include "flandmark_detector.h"

typedef enum {
    fl_center = 0,
    fl_left_eye_int = 1,
    fl_right_eye_int = 2,
    fl_mouth_left = 3,
    fl_mouth_right = 4,
    fl_left_eye_ext = 5,
    fl_right_eye_ext = 6,
    fl_nose = 7,
    fl_count = 8
} fl_indices;

bool ht_fl_estimate(headtracker_t& ctx, Mat& frame, const Rect roi, Mat& rvec_, Mat& tvec_)
{
    int bbox[4];

    bbox[0] = roi.x;
    bbox[1] = roi.y;
#if 0
    bbox[2] = max(roi.width, roi.height) + roi.x;
    bbox[3] = max(roi.width, roi.height) + roi.y;

    if (bbox[0] + bbox[2] > frame.cols)
        bbox[2] = bbox[3] = frame.cols - bbox[0];

    if (bbox[1] + bbox[1] > frame.rows)
        bbox[2] = bbox[3] = frame.rows - bbox[1];
#else
    bbox[2] = roi.width  + roi.x;
    bbox[3] = roi.height + roi.y;
#endif

    IplImage c_image = frame;

    double landmarks[fl_count * 2];

    if (flandmark_detect(&c_image, bbox, ctx.flandmark_model, landmarks))
        return false;

    Point2f left_eye_right = Point2f(
            landmarks[2 * fl_left_eye_int],
            landmarks[2 * fl_left_eye_int + 1]);
    Point2f left_eye_left = Point2f(
            landmarks[2 * fl_left_eye_ext],
            landmarks[2 * fl_left_eye_ext + 1]);
    Point2f right_eye_left = Point2f(
            landmarks[2 * fl_right_eye_int],
            landmarks[2 * fl_right_eye_int + 1]);
    Point2f right_eye_right = Point2f(
            landmarks[2 * fl_right_eye_ext],
            landmarks[2 * fl_right_eye_ext + 1]);
    Point2f nose = Point2f(landmarks[2 * fl_nose], landmarks[2 * fl_nose + 1]);
    Point2f mouth_left = Point2f(
            landmarks[2 * fl_mouth_left],
            landmarks[2 * fl_mouth_left + 1]);
    Point2f mouth_right = Point2f(
			landmarks[2 * fl_mouth_right],
			landmarks[2 * fl_mouth_right + 1]);

    vector<Point2f> image_points(7);
    vector<Point3f> object_points(7);

	object_points[0] = Point3d(-0.03387, -0.03985, 0.14169);
	object_points[1] = Point3d(0.03387, -0.03985, 0.14169);
	object_points[2] = Point3d(-0.08307, -0.04124, 0.1327);
	object_points[3] = Point3d(0.08307, -0.04124, 0.1327);
	object_points[5] = Point3d(-0.04472, 0.08171, 0.16372);
	object_points[6] = Point3d(0.04472, 0.08171, 0.16372);
    object_points[4] = Point3d(0, 0.0335, 0.21822);
    
	for (int i = 0; i < object_points.size(); i++)
	{
		object_points[i].x *= 100;
		object_points[i].y *= 100;
		object_points[i].z *= 100;
	}

    image_points[0] = left_eye_right;
    image_points[1] = right_eye_left;
    image_points[2] = left_eye_left;
    image_points[3] = right_eye_right;
    image_points[5] = mouth_left;
    image_points[6] = mouth_right;
    image_points[4] = nose;
    
    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = ctx.focal_length_w;
    intrinsics.at<float> (1, 1) = ctx.focal_length_h;
    intrinsics.at<float> (0, 2) = ctx.grayscale.cols/2;
    intrinsics.at<float> (1, 2) = ctx.grayscale.rows/2;

    Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);
    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);
    
    for (int i = 0; i < 5; i++)
        dist_coeffs.at<float>(i) = ctx.config.dist_coeffs[i];

    rvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (1, 0) = 1.0;
    
    if (ctx.has_pose)
    {
        rvec = ctx.rvec.clone();
        tvec = ctx.tvec.clone();
    }
    
    if (ctx.has_pose) {
        if (!solvePnP(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec, true, HT_PNP_TYPE))
            return false;
    } else {
        if (!solvePnP(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP))
            return false;
        if (!solvePnP(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec, true, HT_PNP_TYPE))
            return false;
    }

	if (ctx.config.debug && ctx.has_pose)
	{
		vector<Point2f> image_points2;
		projectPoints(object_points, ctx.rvec, ctx.tvec, intrinsics, dist_coeffs, image_points2);
		Scalar color(0, 0, 255);
		float mult = ctx.color.cols / (float)ctx.grayscale.cols;
		Scalar color2(255, 255, 255);
		for (int i = 0; i < image_points.size(); i++)
		{
			line(ctx.color, image_points[i] * mult , image_points2[i] * mult, color, 7);
			circle(ctx.color, image_points[i] * mult, 5, color2, -1);
		}
	}

	rvec_ = rvec.clone();
    tvec_ = tvec.clone();

    return true;
}

bool ht_initial_guess(headtracker_t& ctx, Mat& frame, Mat& rvec_, Mat& tvec_) {
	int ticks = ht_tickcount();

	if (ctx.ticks_last_classification / ctx.config.classification_delay == ticks / ctx.config.classification_delay)
		return false;

	ctx.ticks_last_classification = ticks;

    Rect face_rect;

	if (!ht_classify(ctx.head_classifier, ctx.grayscale, face_rect))
        return false;
    
    if (face_rect.x < 0)
        face_rect.x = 0;
    if (face_rect.y < 0)
        face_rect.y = 0;
    if (face_rect.width + face_rect.x > ctx.grayscale.cols)
        face_rect.width = ctx.grayscale.cols - face_rect.x;
    if (face_rect.height + face_rect.y > ctx.grayscale.rows)
        face_rect.height = ctx.grayscale.rows - face_rect.y;
    
    if (face_rect.width < 10 && face_rect.height < 10)
        return false;

    return ht_fl_estimate(ctx, frame, face_rect, rvec_, tvec_);
}
