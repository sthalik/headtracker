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

bool context::estimate(cv::Mat& frame, const cv::Rect roi, cv::Matx31d& rvec_, cv::Matx31d& tvec_)
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

    if (flandmark_detect(&c_image, bbox, flandmark_model, landmarks))
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

    std::vector<Point2f> image_points(7);
    std::vector<Point3f> object_points(7);

    // XXX TODO numbers still porked
	object_points[0] = Point3d(-0.03387, -0.03985, 0.14169);
	object_points[1] = Point3d(0.03387, -0.03985, 0.14169);
	object_points[2] = Point3d(-0.08307, -0.04124, 0.1327);
	object_points[3] = Point3d(0.08307, -0.04124, 0.1327);
    object_points[5] = Point3d(-0.04472, 0.08171, 0.15877);
    object_points[6] = Point3d(0.04472, 0.08171, 0.15877);
    object_points[4] = Point3d(0, 0.0335, 0.19386);
    
	for (unsigned i = 0; i < object_points.size(); i++)
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
    
    cv::Matx31d rvec, tvec;
    
    if (!cv::solvePnP(object_points, image_points, intrins, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP))
        return false;
    if (!cv::solvePnP(object_points, image_points, intrins, dist, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE))
        return false;
    
    // XXX TODO flandmark ground truth check

	rvec_ = rvec;
    tvec_ = tvec;

    return true;
}

bool context::initial_guess(const cv::Rect rect_, cv::Mat& frame, cv::Matx31d& rvec_, cv::Matx31d& tvec_) {
    Rect rect = rect_;
    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    if (rect.width + rect.x > frame.cols)
        rect.width = frame.cols - rect.x;
    if (rect.height + rect.y > frame.rows)
        rect.height = frame.rows - rect.y;
    
    constexpr int min_size = 30;
    
    if (rect.width < min_size && rect.height < min_size)
        return false;

    return estimate(frame, rect, rvec_, tvec_);
}
