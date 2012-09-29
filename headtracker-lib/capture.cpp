#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool ht_get_image(headtracker_t& ctx) {
    if (!ctx.camera.isOpened())
        return false;

    if (!ctx.camera.read(ctx.color))
        return false;

    cvtColor(ctx.color, ctx.grayscale, CV_BGR2GRAY);
    equalizeHist(ctx.grayscale, ctx.grayscale);
	return true;
}

HT_API(headtracker_t*) ht_make_context(const ht_config_t* config, const char* filename)
{
    headtracker_t* ctx = new headtracker_t;
    if (config == NULL) {
        ht_make_config(&ctx->config);
    } else {
        ctx->config = *config;
    }
    ctx->camera = filename
            ? VideoCapture(filename)
            : VideoCapture(ctx->config.camera_index);

    ctx->classifiers = new classifier_t[HT_CLASSIFIER_COUNT];
	
	ctx->classifiers[HT_CLASSIFIER_HEAD] = ht_make_classifier("haarcascade_frontalface_alt2.xml", ht_make_rect(0, 0, 1, 1), cvSize2D32f(0.1, 0.1));
	ctx->classifiers[HT_CLASSIFIER_EYE1] = ht_make_classifier("haarcascade_lefteye_2splits.xml", ht_make_rect(0.00f, 0.0f, 0.49f, 0.6f), cvSize2D32f(0.1f, 0.05f));
	ctx->classifiers[HT_CLASSIFIER_EYE2] = ht_make_classifier("haarcascade_righteye_2splits.xml", ht_make_rect(0.5f, 0.0f, 0.49f, 0.6f), cvSize2D32f(0.1f, 0.05f));
	ctx->classifiers[HT_CLASSIFIER_NOSE] = ht_make_classifier("haarcascade_mcs_nose.xml", ht_make_rect(0.25f, 0.3f, 0.5f, 0.55f), cvSize2D32f(0.2f, 0.1f));
	ctx->classifiers[HT_CLASSIFIER_MOUTH] = ht_make_classifier("haarcascade_mcs_mouth.xml", ht_make_rect(0.1f, 0.5f, 0.8f, 0.499f), cvSize2D32f(0.2f, 0.1f));

	ctx->ticks_last_classification = ht_tickcount();
	ctx->ticks_last_features = ctx->ticks_last_classification;
	
    ctx->model = ht_load_model("head.raw", cvPoint3D32f(23, 23, 23), cvPoint3D32f(0, 0, 0));
    ctx->keypoint_uv = new CvPoint3D32f[ctx->config.max_keypoints];
	ctx->state = HT_STATE_INITIALIZING;
	ctx->init_retries = 0;
	ctx->restarted = 1;
	ctx->depth_counter_pos = 0;
	ctx->zoom_ratio = 1.0;
	ctx->keypoints = new ht_keypoint[ctx->config.max_keypoints];
	for (int i = 0; i < ctx->config.max_keypoints; i++)
		ctx->keypoints[i].idx = -1;
	ctx->keypoint_count = 0;
    ctx->focal_length = -1;
	if (ctx->config.force_width)
        ctx->camera.set(CV_CAP_PROP_FRAME_WIDTH, ctx->config.force_width);
	if (ctx->config.force_height)
        ctx->camera.set(CV_CAP_PROP_FRAME_HEIGHT, ctx->config.force_height);
    if (ctx->config.force_fps)
        ctx->camera.set(CV_CAP_PROP_FPS, ctx->config.force_fps);
    ctx->abortp = filename != NULL;
    ctx->pyr_a = new vector<Mat>();
    ctx->depth_counter_pos = 0;
    ctx->pyr_b = new vector<Mat>();
    ctx->hz = 0;
    ctx->hz_last_second = -1;
    ctx->ticks_last_second = ht_tickcount() / 1000;
	return ctx;
}

HT_API(void) ht_free_context(headtracker_t* ctx) {
	if (ctx->keypoint_uv)
		delete[] ctx->keypoint_uv;
	if (ctx->model.triangles)
		delete[] ctx->model.triangles;
	if (ctx->model.projection)
		delete[] ctx->model.projection;
	if (ctx->model.centers)
		delete[] ctx->model.centers;
    if (ctx->model.projected_depths)
        delete[] ctx->model.projected_depths;
	if (ctx->keypoints)
		delete[] ctx->keypoints;
    delete ctx->pyr_a;
    delete ctx->pyr_b;
    delete[] ctx->classifiers;
	delete ctx;
}

HT_API(void) ht_get_bgr_frame(headtracker_t* ctx, ht_frame_t* ret) {
    ret->cols = ctx->color.cols;
    ret->rows = ctx->color.rows;
    ret->channels = ctx->color.channels();

    if (ret->cols > 0) {
        ret->data = new unsigned char[ret->cols * ret->rows * ret->channels];
        memcpy(ret->data, ctx->color.data, ret->cols * ret->rows * ret->channels);
    } else {
        ret->data = NULL;
    }
}
