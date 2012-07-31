// todo die on impossible poses
// todo autocalibration
// todo do away with leaks if initialization fails
#pragma once
#define HT_PI 3.14159265f
#define HT_STD_DEPTH 560.0f

#define HT_CENTROID_DEPTH 100.0f
#define HT_CENTROID_Y 25.0f

#define HT_FEATURE_MAX_QUALITY_LEVEL 60
#define HT_FEATURE_MIN_QUALITY_LEVEL 1

// these ones will be trainable
// maybe even more after training.cpp is written and sample video(s) made
// the basic idea is to make a video of doing something(s) the tracker has
// problems with, then change random parameters until it's more fit
// the fitness is problematic, but will probably be measured by
// how a freshly-computed pose differs from the 'continuing'
// also, the reprojection error of the present pose
// these will be probably tested later, but are pretty optimal as of now

typedef enum {
	HT_STATE_INITIALIZING = 0, // waiting for RANSAC consensus
	HT_STATE_TRACKING = 1, // ransac consensus established
	HT_STATE_LOST = 2, // consensus lost; fall back to initializing
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
	CvHaarClassifierCascade* cascade;
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
	int count;
	CvPoint3D32f* centers;
} model_t;

static __inline float ht_dot_product2d(CvPoint2D32f point1, CvPoint2D32f point2) {
	return point1.x * point2.x + point1.y * point2.y;
}

static __inline int ht_tickcount(void) {
	return (int) (cv::getTickCount() * 1000 / cv::getTickFrequency());
}

static __inline CvPoint2D32f ht_project_point(CvPoint3D32f point, float* rotation_matrix, float* translation_vector, float focal_length) {
	float x = point.x * rotation_matrix[0] + point.y * rotation_matrix[1] + point.z * rotation_matrix[2] + translation_vector[0];
	float y = point.x * rotation_matrix[3] + point.y * rotation_matrix[4] + point.z * rotation_matrix[5] + translation_vector[1];
	float z = point.x * rotation_matrix[6] + point.y * rotation_matrix[7] + point.z * rotation_matrix[8] + translation_vector[2];

	focal_length /= z;

	return cvPoint2D32f(x * focal_length,
						y * focal_length);
}

typedef struct {
	int idx;
	CvPoint2D32f position;
} ht_keypoint;

typedef struct ht_context {
	CvCapture* camera;
	IplImage* grayscale;
	IplImage* color;
	classifier_t* classifiers;
	int ticks_last_classification;
	int ticks_last_features;
	model_t model;
	state_t state;
	CvPoint2D32f* features;
	char* feature_failed_iters;
	IplImage* pyr_a;
	IplImage* pyr_b;
	IplImage* last_image;
	int feature_count;
	int init_retries;
	bool restarted;
	float* depths;
	int depth_frame_count;
	unsigned char depth_counter_pos;
	float zoom_ratio;
	ht_config_t config;
	char* bgr_frame;
	ht_keypoint* keypoints;
	int keypoint_count;
	char* keypoint_failed_iters;
} headtracker_t;

model_t ht_load_model(const char* filename, CvPoint3D32f scale, CvPoint3D32f offset);
void ht_free_model(model_t& model);
CvPoint2D32f ht_project_point(CvPoint3D32f point, float* rotation_matrix, float* translation_vector, float focal_length);
CvPoint2D32f ht_point_to_2d(CvPoint3D32f point);
bool ht_point_inside_triangle_2d(CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c, CvPoint2D32f point);

bool ht_posit(CvPoint2D32f* image_points, CvPoint3D32f* model_points, int point_cnt, float* rotation_matrix, float* translation_vector, CvTermCriteria term_crit, float focal_length);

classifier_t ht_make_classifier(const char* filename, rect_t rect, CvSize2D32f min_size);
bool ht_classify(headtracker_t& ctx, const classifier_t& classifier, IplImage& frame, const CvRect& roi, CvRect& ret);
void ht_free_classifier(classifier_t* classifier);

typedef enum {
	HT_CLASSIFIER_HEAD = 0,
	HT_CLASSIFIER_NOSE = 1,
	HT_CLASSIFIER_EYE1 = 2,
	HT_CLASSIFIER_EYE2 = 3,
	HT_CLASSIFIER_MOUTH = 4,
	HT_CLASSIFIER_COUNT = 5
} classifiers_t;

bool ht_get_image(headtracker_t& ctx);

bool ht_initial_guess(headtracker_t& ctx, IplImage& frame, float* rotation_matrix, float* translation_vector);
ht_result_t ht_matrix_to_euler(float* rotation_matrix, float* translation_vector);
bool ht_point_inside_rectangle(CvPoint2D32f p, CvPoint2D32f topLeft, CvPoint2D32f bottomRight);
void ht_project_model(headtracker_t& ctx,
					  float* rotation_matrix,
					  float* translation_vector,
					  model_t& model,
					  CvPoint3D32f origin);
bool ht_triangle_at(headtracker_t& ctx, CvPoint2D32f pos, triangle_t* ret, int* idx, float* rotation_matrix, float* translation_vector, const model_t& model);
bool ht_triangle_exists(headtracker_t& ctx, CvPoint2D32f pos, const model_t& model);
void ht_draw_model(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model);
void ht_get_features(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model);
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

typedef struct {
	float avg;
} error_t;

bool ht_ransac(headtracker_t& ctx,
			   int max_iter,
			   float max_error,
			   int min_consensus,
			   int* best_feature_cnt,
			   int* best_keypoint_cnt,
			   error_t* best_error,
			   int* best_indices,
			   int* best_keypoints,
			   model_t& model,
			   float error_scale);



bool ht_estimate_pose(headtracker_t& ctx,
					  float* rotation_matrix,
					  float* translation_vector,
					  float* rotation_matrix2,
					  float* translation_vector2,
					  CvPoint3D32f* offset,
					  CvPoint2D32f* image_centroid);
bool ht_ransac_best_indices(headtracker_t& ctx, error_t* best_error);
void ht_remove_lumps(headtracker_t& ctx);
void ht_update_zoom_scale(headtracker_t& ctx, float translation_2);

