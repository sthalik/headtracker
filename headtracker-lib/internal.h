// todo die on impossible poses
// todo do away with leaks if initialization fails
#pragma once
#include <vector>
using namespace std;
using namespace cv;
#include "common.h"
#include <opencv2/opencv.hpp>
#define HT_PI 3.1415926535
#define HT_STD_DEPTH 500.0f

#define HT_FEATURE_MAX_QUALITY_LEVEL 60
#define HT_FEATURE_MIN_QUALITY_LEVEL 8

typedef enum {
	HT_STATE_INITIALIZING = 0, // waiting for RANSAC consensus
	HT_STATE_TRACKING = 1, // ransac consensus established
    HT_STATE_LOST = 2 // consensus lost; fall back to initializing
} state_t;

typedef struct {
	float x, y, w, h;
} rect_t;

static __inline rect_t ht_make_rect(float x, float y, float w, float h) {
	rect_t ret;

	ret.x = x;
	ret.y = y;
	ret.w = w;
	ret.h = h;

	return ret;
}

typedef struct {
	rect_t rect;
	CvSize2D32f min_size;
    CascadeClassifier cascade;
} classifier_t;

typedef struct {
    CvPoint3D32f p1;
    CvPoint3D32f p2;
    CvPoint3D32f p3;
} triangle_t;

typedef struct {
    CvPoint2D32f p1;
    CvPoint2D32f p2;
    CvPoint2D32f p3;
} triangle2d_t;

typedef struct {
	triangle_t* triangles;
	triangle2d_t* projection;
	float* projected_depths;
	int count;
	CvPoint3D32f* centers;
} model_t;

static __inline float ht_dot_product2d(CvPoint2D32f point1, CvPoint2D32f point2) {
	return point1.x * point2.x + point1.y * point2.y;
}

static __inline int ht_tickcount(void) {
	return (int) (cv::getTickCount() * 1000 / cv::getTickFrequency());
}

static __inline CvPoint2D32f ht_project_point(CvPoint3D32f point, const float* rotation_matrix, const float* translation_vector, const float f) {
    float x = point.x * rotation_matrix[0] + point.y * rotation_matrix[1] + point.z * rotation_matrix[2] + translation_vector[0];
    float y = point.x * rotation_matrix[3] + point.y * rotation_matrix[4] + point.z * rotation_matrix[5] + translation_vector[1];
    float z = point.x * rotation_matrix[6] + point.y * rotation_matrix[7] + point.z * rotation_matrix[8] + translation_vector[2];

    return cvPoint2D32f(x * f / z,
						y * f / z);
}

typedef struct {
	int idx;
	CvPoint2D32f position;
    int frames;
} ht_keypoint;

typedef struct ht_context {
    float focal_length;
    float focal_length_w;
    float focal_length_h;
    VideoCapture camera;
    Mat grayscale;
    Mat color;
	classifier_t* classifiers;
	int ticks_last_classification;
	int ticks_last_features;
	model_t model;
	state_t state;
    vector<Mat>* pyr_a;
    vector<Mat>* pyr_b;
    Mat last_image;
    int init_retries;
	bool restarted;
	unsigned char depth_counter_pos;
	float zoom_ratio;
	ht_config_t config;
	ht_keypoint* keypoints;
    int keypoint_count;
    CvPoint3D32f* keypoint_uv;
    int ticks_last_second;
    int hz;
    int hz_last_second;
    bool abortp;
    Mat face_histogram;
    Mat tmp;
} headtracker_t;

HT_API(void) ht_reset(headtracker_t* ctx);

model_t ht_load_model(const char* filename, CvPoint3D32f scale, CvPoint3D32f offset);
void ht_free_model(model_t& model);
CvPoint2D32f ht_point_to_2d(CvPoint3D32f point);
bool ht_point_inside_triangle_2d(const CvPoint2D32f a, const CvPoint2D32f b, const CvPoint2D32f c, const CvPoint2D32f point, CvPoint2D32f& uv);

bool ht_posit(CvPoint2D32f* image_points, CvPoint3D32f* model_points, int point_cnt, float* rotation_matrix, float* translation_vector, CvTermCriteria term_crit, float focal_length);

classifier_t ht_make_classifier(const char* filename, rect_t rect, CvSize2D32f min_size);
bool ht_classify(classifier_t& classifier, Mat& frame, const Rect& roi, Rect& ret);

typedef enum {
	HT_CLASSIFIER_HEAD = 0,
    HT_CLASSIFIER_NOSE = 1,
    HT_CLASSIFIER_EYE1 = 2,
    HT_CLASSIFIER_EYE2 = 3,
    HT_CLASSIFIER_MOUTH = 4,
    HT_CLASSIFIER_COUNT = 5
} classifiers_t;

bool ht_get_image(headtracker_t& ctx);

bool ht_initial_guess(headtracker_t& ctx, Mat& frame, float *rotation_matrix, float *translation_vector);
ht_result_t ht_matrix_to_euler(float *rotation_matrix, float *translation_vector);
bool ht_point_inside_rectangle(CvPoint2D32f p, CvPoint2D32f topLeft, CvPoint2D32f bottomRight);
void ht_project_model(headtracker_t& ctx,
                      float *rotation_matrix,
                      float *translation_vector,
                      model_t& model,
                      CvPoint3D32f origin);
bool ht_triangle_at(const CvPoint2D32f pos, triangle_t* ret, int* idx, const model_t& model, CvPoint2D32f& uv);
bool ht_triangle_exists(CvPoint2D32f pos, const model_t& model);
void ht_draw_model(headtracker_t& ctx, model_t& model);
void ht_get_features(headtracker_t& ctx, model_t& model);
void ht_track_features(headtracker_t& ctx);
void ht_draw_features(headtracker_t& ctx);

static __inline float ht_distance2d_squared(CvPoint2D32f p1, CvPoint2D32f p2) {
	float x = p1.x - p2.x;
	float y = p1.y - p2.y;
	return x * x + y * y;
}

static __inline float ht_distance3d_squared(CvPoint3D32f p1, CvPoint3D32f p2) {
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

bool ht_estimate_pose(headtracker_t& ctx,
                      float *rotation_matrix,
                      float *translation_vector, float *rotation_matrix2, float *translation_vector2,
                      CvPoint3D32f* offset);
bool ht_ransac_best_indices(headtracker_t& ctx, float *best_error);
void ht_update_zoom_scale(headtracker_t& ctx, float translation_2);
CvPoint3D32f ht_get_triangle_pos(const CvPoint2D32f uv, const triangle_t& t);
void ht_remove_outliers(headtracker_t& ctx);
CvRect ht_get_roi(const headtracker_t& ctx, model_t& model);

