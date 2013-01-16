#include "stdafx.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool ht_get_image(headtracker_t& ctx) {
    if (!ctx.camera.isOpened())
        return false;

    if (!ctx.camera.read(ctx.color))
        return false;

    //Mat tmp;

    //ctx.color.copyTo(tmp);
    //resize(ctx.color, tmp, Size(320, 320 * ctx.color.rows / ctx.color.cols), 0, 0, CV_INTER_AREA);

    //ctx.color = tmp;

    cvtColor(ctx.color, ctx.grayscale, CV_BGR2GRAY);
    ctx.grayscale.copyTo(ctx.tmp);
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

    ctx->head_classifier = CascadeClassifier("haarcascade_frontalface_alt2.xml");

	ctx->ticks_last_classification = ht_tickcount();
	ctx->ticks_last_features = ctx->ticks_last_classification;
	
    ctx->model = ht_load_model("head.raw");
    ctx->keypoint_uv = new Point3f[ctx->config.max_keypoints];
	ctx->state = HT_STATE_INITIALIZING;
    ctx->restarted = true;
	ctx->zoom_ratio = 1.0;
    ctx->keypoints = new ht_keypoint[ctx->config.max_keypoints];
    for (int i = 0; i < ctx->config.max_keypoints; i++)
		ctx->keypoints[i].idx = -1;
    ctx->keypoint_count = 0;
    ctx->focal_length_w = -1;
    ctx->focal_length_h = -1;
	if (ctx->config.force_width)
        ctx->camera.set(CV_CAP_PROP_FRAME_WIDTH, ctx->config.force_width);
	if (ctx->config.force_height)
        ctx->camera.set(CV_CAP_PROP_FRAME_HEIGHT, ctx->config.force_height);
    if (ctx->config.force_fps)
        ctx->camera.set(CV_CAP_PROP_FPS, ctx->config.force_fps);
    ctx->pyr_a = new vector<Mat>();
    ctx->pyr_b = new vector<Mat>();
    ctx->hz = 0;
    ctx->hz_last_second = -1;
    ctx->ticks_last_second = ht_tickcount() / 1000;
    ctx->has_pose = false;
    ctx->flandmark_model = flandmark_init("flandmark_model.dat");
	return ctx;
}

HT_API(void) ht_free_context(headtracker_t* ctx) {
	if (ctx->keypoint_uv)
		delete[] ctx->keypoint_uv;
	if (ctx->model.triangles)
		delete[] ctx->model.triangles;
	if (ctx->model.projection)
		delete[] ctx->model.projection;
	if (ctx->keypoints)
		delete[] ctx->keypoints;
    delete ctx->pyr_a;
    delete ctx->pyr_b;
    flandmark_free(ctx->flandmark_model);
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
