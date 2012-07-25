#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_remove_lumps(headtracker_t& ctx) {
	if (ctx.feature_count > HT_MAX_TRACKED_FEATURES * HT_FILTER_LUMPS_FEATURE_COUNT_THRESHOLD) {
		for (int i = 0; i < ctx.tracking_model.count; i++) {
			if (ctx.features[i].x == -1 || ctx.feature_failed_iters == 0)
				continue;
			for (int j = 0; j < ctx.tracking_model.count; j++) {
				if (i == j)
					continue;
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < HT_MIN_POINT_DISTANCE * HT_FILTER_LUMPS_DISTANCE_THRESHOLD) {
					if (ctx.feature_failed_iters[i] >= ctx.feature_failed_iters[j]) {
						ctx.features[i] = cvPoint2D32f(-1, -1);
						ctx.feature_failed_iters[i] = 0;
					} else {
						ctx.features[j] = cvPoint2D32f(-1, -1);
						ctx.feature_failed_iters[j] = 0;
					}
					ctx.feature_count--;
				}
			}
		}
	}
	if (ctx.feature_count > HT_MAX_TRACKED_FEATURES * HT_FILTER_LUMPS_FEATURE_COUNT_THRESHOLD) {
		for (int i = 0; i < ctx.tracking_model.count; i++) {
			if (ctx.features[i].x == -1)
				continue;
			for (int j = 0; j < ctx.tracking_model.count; j++) {
				if (i == j)
					continue;
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < HT_MIN_POINT_DISTANCE * HT_FILTER_LUMPS_DISTANCE_THRESHOLD) {
					ctx.features[i] = cvPoint2D32f(-1, -1);
					ctx.feature_failed_iters[i] = 0;
					ctx.feature_count--;
				}
			}
		}
	}
}

void ht_draw_features(headtracker_t& ctx) {
	if (!ctx.features)
		return;

	for (int i = 0; i < ctx.tracking_model.count; i++) {
		if (ctx.features[i].x == -1 || ctx.features[i].y == -1)
			continue;

		CvScalar color;
		int size;

		if (ctx.feature_failed_iters[i] == 0) {
			color = CV_RGB(0, 255, 255);
			size = 1;
		} else if (ctx.feature_failed_iters[i] == 1) {
			size = 3;
			color = CV_RGB(255, 255, 0);
		} else {
			color = CV_RGB(255, 0, 0);
			size = 4;
		}

		cvCircle(ctx.color, cvPoint(ctx.features[i].x, ctx.features[i].y), size, color, -1);
	}
}

void ht_track_features(headtracker_t& ctx) {
	if (!ctx.features)
		return;

	IplImage* frame = ctx.grayscale;

	if (!ctx.last_image)
		return;

	int got_pyr = ctx.pyr_a != NULL;

	if (!ctx.pyr_a) {
		CvSize pyr_size = cvSize(ctx.grayscale->width+8, ctx.grayscale->height/3);
		ctx.pyr_a = cvCreateImage(pyr_size, IPL_DEPTH_32F, 1);
		ctx.pyr_b = cvCreateImage(pyr_size, IPL_DEPTH_32F, 1);
	}

	int sz = 0, max = ctx.tracking_model.count;
	CvPoint2D32f* old_features = new CvPoint2D32f[ctx.tracking_model.count];

	for (int i = 0; i < max; i++) {
		if (ctx.features[i].x != -1 || ctx.features[i].y != -1)
			old_features[sz++] = ctx.features[i];
	}

	if (sz == 0) {
		delete[] old_features;
		return;
	}

	char* features_found = new char[ctx.tracking_model.count];
	CvPoint2D32f* new_features = new CvPoint2D32f[ctx.tracking_model.count];
	
	cvCalcOpticalFlowPyrLK(
		ctx.last_image,
		ctx.grayscale,
		ctx.pyr_a,
		ctx.pyr_b,
		old_features,
		new_features,
		sz,
		cvSize(HT_PYRLK_WIN_SIZE, HT_PYRLK_WIN_SIZE),
		HT_PYRLK_PYRAMIDS,
		features_found,
		NULL,
		cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.3 ),
		(got_pyr && !ctx.restarted) ? CV_LKFLOW_PYR_A_READY : 0);

	ctx.restarted = 0;

	for (int i = 0, j = 0; i < sz; i++, j++) {
		for (; j < ctx.tracking_model.count; j++)
			if (ctx.features[j].x != -1 || ctx.features[j].y != -1)
				break;

		if (j == ctx.tracking_model.count)
			break;

		if (!features_found[i]) {
			ctx.features[j] = cvPoint2D32f(-1, -1);
			ctx.feature_count--;
			ctx.feature_failed_iters[j] = 0;
		} else
			ctx.features[j] = new_features[i];
	}

	IplImage* tmp = ctx.pyr_a;
	ctx.pyr_a = ctx.pyr_b;
	ctx.pyr_b = tmp;

	delete[] features_found;
	delete[] new_features;
	delete[] old_features;
}

void ht_get_features(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model, CvPoint3D32f origin) {
	if (ctx.feature_count >= HT_MAX_TRACKED_FEATURES * HT_DETECT_FEATURES_THRESHOLD)
		return;

	if (!model.projection)
		return;

	float min_x = ctx.grayscale->width, max_x = 0;
	float min_y = ctx.grayscale->height, max_y = 0;

	int sz = model.count;

	if (!ctx.features) {
		ctx.features = new CvPoint2D32f[sz];
		for (int i = 0; i < sz; i++)
			ctx.features[i] = cvPoint2D32f(-1, -1);
		ctx.feature_failed_iters = new char[sz];
		for (int i = 0; i < sz; i++)
			ctx.feature_failed_iters[i] = 0;
	}

	for (int i = 0; i < ctx.tracking_model.count; i++) {
		float x = (model.projection[i].p1.x + model.projection[i].p2.x + model.projection[i].p3.x) / 3;
		float y = (model.projection[i].p1.x + model.projection[i].p2.y + model.projection[i].p3.y) / 3;
		if (x > max_x)
			max_x = x;
		if (x < min_x)
			min_x = x;
		if (y > max_y)
			max_y = y;
		if (y < min_y)
			min_y = y;
	}

	CvRect roi = cvRect(min_x, min_y, max_x - min_x, max_y - min_y);

	roi.x = max(0, min(roi.x, ctx.grayscale->width - 1));
	roi.y = max(0, min(roi.y, ctx.grayscale->height - 1));
	roi.width = max(0, min(ctx.grayscale->width - roi.x, roi.width));
	roi.height = max(0, min(ctx.grayscale->height - roi.y, roi.height));

	if (roi.width == 0 || roi.height == 0)
		return;

	IplImage* eig_image = cvCreateImage( cvGetSize(ctx.grayscale), IPL_DEPTH_32F, 1 );
	IplImage* tmp_image = cvCreateImage( cvGetSize(ctx.grayscale), IPL_DEPTH_32F, 1 );

	CvPoint2D32f* tmp_features    = new CvPoint2D32f[HT_MAX_DETECT_FEATURES];
	CvPoint2D32f* features_to_add = new CvPoint2D32f[HT_MAX_DETECT_FEATURES];
	int k = 0;

	int cnt = HT_MAX_DETECT_FEATURES;

	if (cnt > 0) {
		cvSetImageROI(ctx.grayscale, roi);
		cvGoodFeaturesToTrack(ctx.grayscale, eig_image, tmp_image, tmp_features, &cnt, HT_FEATURE_QUALITY_LEVEL, HT_DETECT_POINT_DISTANCE, NULL, 3, HT_USE_HARRIS);
		cvResetImageROI(ctx.grayscale);
	}

	for (int i = 0; i < cnt; i++) {
		tmp_features[i].x += roi.x;
		tmp_features[i].y += roi.y;
		
		triangle_t t;
		int idx;

		if (!(ht_triangle_at(ctx, cvPoint(tmp_features[i].x, tmp_features[i].y), &t, &idx, rotation_matrix, translation_vector, model)))
			continue;

		if (ctx.features[idx].x != -1 || ctx.features[idx].y != -1)
			continue;

		features_to_add[k++] = tmp_features[i];
	}

	if (k > 0 && roi.width > 32 && roi.height > 32)
		cvFindCornerSubPix(ctx.grayscale, features_to_add, k, cvSize(15, 15), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.2));

	for (int i = 0; i < k && ctx.feature_count < HT_MAX_TRACKED_FEATURES; i++) {
		triangle_t t;
		int idx;

		if (!(ht_triangle_at(ctx, cvPoint(features_to_add[i].x, features_to_add[i].y), &t, &idx, rotation_matrix, translation_vector, model)))
			continue;

		if (ctx.features[idx].x != -1 || ctx.features[idx].y != -1)
			continue;

		for (int j = 0; j < model.count; j++) {
			if (ctx.features[j].x != -1 && ht_distance2d_squared(features_to_add[i], ctx.features[j]) < HT_MIN_POINT_DISTANCE * HT_MIN_POINT_DISTANCE)
				goto end2;
		}

		ctx.features[idx] = features_to_add[i];
		ctx.feature_count++;
		//ht_project_model(ctx, rotation_matrix, translation_vector, model, origin);

end2:
		;
	}

	cvReleaseImage(&eig_image);
	cvReleaseImage(&tmp_image);
	delete tmp_features;
	delete features_to_add;
}
