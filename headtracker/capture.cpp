#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_get_image(headtracker_t& ctx) {
	IplImage* frame = cvQueryFrame(ctx.camera);

	if (frame == NULL)
		return false;

	ctx.color = frame;

	if (ctx.grayscale == NULL)
		ctx.grayscale = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);

	cvCvtColor(frame, ctx.grayscale, CV_BGR2GRAY);
	cvEqualizeHist(ctx.grayscale, ctx.grayscale);

	return true;
}

headtracker_t* ht_make_context(int camera_idx) {
	headtracker_t* ctx = new headtracker_t;

	ctx->grayscale = NULL;
	ctx->camera = cvCreateCameraCapture(camera_idx);
	ctx->classifiers = new classifier_t[HT_CLASSIFIER_COUNT];
	ctx->color = NULL;
	
	ctx->classifiers[HT_CLASSIFIER_HEAD] = ht_make_classifier("haarcascade_frontalface_alt2.xml", ht_make_rect(0, 0, 1, 1), cvSize2D32f(0.1, 0.1));
	ctx->classifiers[HT_CLASSIFIER_EYE1] = ht_make_classifier("haarcascade_lefteye_2splits.xml", ht_make_rect(0.08f, 0.15f, 0.38f, 0.4f), cvSize2D32f(0.15f, 0.10f));
	ctx->classifiers[HT_CLASSIFIER_EYE2] = ht_make_classifier("haarcascade_righteye_2splits.xml", ht_make_rect(0.58f, 0.15f, 0.38f, 0.4f), cvSize2D32f(0.15f, 0.10f));
	ctx->classifiers[HT_CLASSIFIER_NOSE] = ht_make_classifier("haarcascade_mcs_nose.xml", ht_make_rect(0.33f, 0.35f, 0.34f, 0.4f), cvSize2D32f(0.2f, 0.1f));
	ctx->classifiers[HT_CLASSIFIER_MOUTH] = ht_make_classifier("haarcascade_mcs_mouth.xml", ht_make_rect(0.15f, 0.6f, 0.7f, 0.39f), cvSize2D32f(0.3f, 0.1f));

	ctx->ticks_last_classification = GetTickCount();
	ctx->ticks_last_features = ctx->ticks_last_classification;
	
	ctx->model = ht_load_model("head.raw", cvPoint3D32f(1, 1, 1), cvPoint3D32f(0, 0, 0));
	ctx->features = NULL;
	ctx->pyr_a = NULL;
	ctx->pyr_b = NULL;
	ctx->last_image = NULL;
	ctx->feature_count = 0;
	ctx->state = HT_STATE_INITIALIZING;
	ctx->mouse_x = ctx->mouse_y = -1;
	ctx->init_retries = 0;
	ctx->restarted = 1;
	for (int i = 0; i < HT_DEPTH_AVG_FRAMES; i++)
		ctx->depths[i] = 0;
	ctx->depth_frame_count = 0;
	ctx->depth_counter_pos = 0;
	ctx->zoom_ratio = 1.0;
	return ctx;
}

void ht_free_context(headtracker_t* ctx) {
	if (ctx->model.triangles)
		delete ctx->model.triangles;
	if (ctx->model.projection)
		delete ctx->model.projection;
	if (ctx->model.centers)
		delete ctx->model.centers;
	cvReleaseCapture(&ctx->camera);
	for (int i = 0; i < HT_CLASSIFIER_COUNT; i++)
		ht_free_classifier(&ctx->classifiers[i]);
	if (ctx->grayscale)
		cvReleaseImage(&ctx->grayscale);
	if (ctx->features)
		delete ctx->features;
	if (ctx->pyr_a)
		cvReleaseImage(&ctx->pyr_a);
	if (ctx->pyr_b)
		cvReleaseImage(&ctx->pyr_b);
	if (ctx->last_image)
		cvReleaseImage(&ctx->last_image);
	delete ctx;
}