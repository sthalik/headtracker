#pragma once
#define HT_FOCAL_LENGTH 602
#define HT_MODEL_X_SCALE 28.846153
#define HT_MODEL_Y_SCALE 28.846153
#define HT_MODEL_Z_SCALE 28.846153
#define HT_CLASSIFICATION_DELAY_MS 200
#define HT_PI 3.14159265
#define HT_FEATURE_QUALITY_LEVEL 0.004
#define HT_PYRLK_PYRAMIDS 4
#define HT_PYRLK_WIN_SIZE 12

#define HT_MAX_TRACKED_FEATURES 50

#define HT_MODEL_MIN_DEPTH 2.8

#define HT_HAAR_MODEL_Y_OFFSET 0
#define HT_HAAR_MODEL_Z_OFFSET -4.1904761904761904761904761904762

#define HT_MODEL_Y_OFFSET 0
#define HT_MODEL_Z_OFFSET -4.1904761904761904761904761904762

#define HT_RANSAC_MIN_CONSENSUS 15
#define HT_RANSAC_MAX_ERROR 0.92f
#define HT_RANSAC_ITER 100
#define HT_RANSAC_MIN_POINTS 7
#define HT_RANSAC_MAX_BEST_ERROR 13.0
#define HT_RANSAC_AVG_BEST_ERROR 4.8

#define HT_MIN_POINT_DISTANCE 20.01
#define HT_DETECT_POINT_DISTANCE 5.01

#define HT_MAX_DETECT_FEATURES 600
#define HT_MIN_TRACK_START_POINTS 25

#define HT_MAX_INIT_RETRIES 30

typedef enum {
	HT_STATE_INITIALIZING = 0, // waiting for RANSAC consensus
	HT_STATE_TRACKING = 1, // ransac consensus established
	HT_STATE_LOST = 2, // consensus lost; fall back to initializing
} state_t;

typedef struct {
	double x, y, w, h;
} rect_t;

static __inline rect_t ht_make_rect(double x, double y, double w, double h) {
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
	float rotx, roty, rotz;
	float tx, ty, tz;
} euler_t;

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
} model_t;

static __inline float ht_dot_product2d(CvPoint2D32f point1, CvPoint2D32f point2) {
	return point1.x * point2.x + point1.y * point2.y;
}

typedef struct {
	CvCapture* camera;
	IplImage* grayscale;
	IplImage* color;
	classifier_t* classifiers;
	int ticks_last_classification;
	int ticks_last_features;
	model_t tracking_model;
	model_t haar_model;
	state_t state;
	int mouse_x, mouse_y;
	CvPoint2D32f* features;
	IplImage* pyr_a;
	IplImage* pyr_b;
	IplImage* last_image;
	int feature_count;
	int init_retries;
	bool restarted;
} headtracker_t;

model_t ht_load_model(const char* filename, CvPoint3D64f scale, CvPoint3D64f offset, float max_z);
void ht_free_model(model_t& model);
CvPoint3D32f ht_project_point(CvPoint3D32f point, float* rotation_matrix, float* translation_vector);
CvPoint2D32f ht_point_to_2d(CvPoint3D32f point);
bool ht_point_inside_triangle_2d(CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c, CvPoint2D32f point);

bool ht_posit(CvPoint2D32f* image_points, CvPoint3D32f* model_points, int point_cnt, float* rotation_matrix, float* translation_vector, CvTermCriteria term_crit = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 25, 0.015));

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
headtracker_t* ht_make_context(int camera_idx);
void ht_free_context(headtracker_t* ctx);

bool ht_initial_guess(headtracker_t& ctx, IplImage& frame, float* rotation_matrix, float* translation_vector);
euler_t ht_matrix_to_euler(float* rotation_matrix, float* translation_vector);
bool ht_point_inside_rectangle(CvPoint2D32f p, CvPoint2D32f topLeft, CvPoint2D32f bottomRight);
CvPoint2D32f ht_point_to_screen(CvPoint3D32f p, float* rotation_matrix, float* translation_vector);
void ht_project_model(headtracker_t& ctx,
					  float* rotation_matrix,
					  float* translation_vector,
					  model_t& model,
					  CvPoint3D32f origin);
bool ht_triangle_at(headtracker_t& ctx, CvPoint pos, triangle_t* ret, int* idx, float* rotation_matrix, float* translation_vector, model_t& model);
void ht_draw_model(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model);
void ht_get_features(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model, CvPoint3D32f offset);
void ht_track_features(headtracker_t& ctx);
void ht_draw_features(headtracker_t& ctx);

static __inline float ht_distance2d_squared(CvPoint2D32f p1, CvPoint2D32f p2) {
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

static __inline float ht_distance3d_squared(CvPoint3D32f p1, CvPoint3D32f p2) {
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

typedef struct {
	float max;
	float avg;
} error_t;

error_t ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt);

bool ht_ransac(headtracker_t& ctx,
			   int max_iter,
			   int iter_points,
			   float max_error,
			   int min_consensus,
			   int* best_cnt,
			   error_t* best_error,
			   int* best_indices,
			   model_t& model);

bool ht_estimate_pose(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, int* indices, int count, CvPoint3D32f* offset);
bool ht_ransac_best_indices(headtracker_t& ctx, int* best_cnt, error_t* best_error, int* best_indices);