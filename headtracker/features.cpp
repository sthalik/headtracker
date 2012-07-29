#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_remove_lumps(headtracker_t& ctx) {
	float max = ctx.config.min_feature_distance * ctx.config.filter_lumps_distance_threshold * ctx.zoom_ratio;
	float threshold = ctx.config.max_tracked_features * ctx.config.filter_lumps_feature_count_threshold;
	if (ctx.feature_count > threshold) {
		for (int i = 0; i < ctx.model.count; i++) {
			if (ctx.features[i].x == -1 || ctx.feature_failed_iters == 0)
				continue;
			for (int j = 0; j < ctx.model.count; j++) {
				if (i == j)
					continue;
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < max) {
					int idx = ctx.feature_failed_iters[i] >= ctx.feature_failed_iters[j]
									? i
									: j;
					ctx.features[idx] = cvPoint2D32f(-1, -1);
					ctx.feature_failed_iters[idx] = 0;
					ctx.feature_count--;
				}
			}
		}
	}
	if (ctx.feature_count > threshold) {
		for (int i = 0; i < ctx.model.count; i++) {
			if (ctx.features[i].x == -1)
				continue;
			for (int j = 0; j < ctx.model.count; j++) {
				if (i == j)
					continue;
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < max) {
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

	for (int i = 0; i < ctx.model.count; i++) {
		if (ctx.features[i].x == -1 || ctx.features[i].y == -1)
			continue;

		CvScalar color;

		if (ctx.feature_failed_iters[i] == 0) {
			color = CV_RGB(0, 255, 255);
		} else if (ctx.feature_failed_iters[i] < ctx.config.feature_max_failed_ransac) {
			color = CV_RGB(255, 255, 0);
		} else {
			color = CV_RGB(255, 0, 0);
		}

		cvCircle(ctx.color, cvPoint(ctx.features[i].x, ctx.features[i].y), 1, color, -1);
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

	int sz = 0, max = ctx.model.count;
	CvPoint2D32f* old_features = new CvPoint2D32f[ctx.model.count];

	for (int i = 0; i < max; i++) {
		if (ctx.features[i].x != -1 || ctx.features[i].y != -1)
			old_features[sz++] = ctx.features[i];
	}

	if (sz == 0) {
		delete[] old_features;
		return;
	}

	char* features_found = new char[sz];
	CvPoint2D32f* new_features = new CvPoint2D32f[sz];
	
	cvCalcOpticalFlowPyrLK(
		ctx.last_image,
		ctx.grayscale,
		ctx.pyr_a,
		ctx.pyr_b,
		old_features,
		new_features,
		sz,
		cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
		ctx.config.pyrlk_pyramids,
		features_found,
		NULL,
		cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 200, 0.1),
		OPTFLOW_LK_GET_MIN_EIGENVALS | (got_pyr && !ctx.restarted) ? CV_LKFLOW_PYR_A_READY : 0);

	for (int i = 0, j = 0; i < sz; i++, j++) {
		for (; j < ctx.model.count; j++)
			if (ctx.features[j].x != -1 || ctx.features[j].y != -1)
				break;

		if (j == ctx.model.count)
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

static bool ht_feature_quality_level(const KeyPoint x, const KeyPoint y) {
	return x.response < y.response;
}

void ht_get_features(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model) {
	if (ctx.feature_count >= ctx.config.max_tracked_features * ctx.config.features_detect_threshold)
		return;

	if (!model.projection)
		return;

	float min_x = (float) ctx.grayscale->width, max_x = 0.0f;
	float min_y = (float) ctx.grayscale->height, max_y = 0.0f;

	int sz = model.count;

	if (!ctx.features) {
		ctx.features = new CvPoint2D32f[sz];
		for (int i = 0; i < sz; i++)
			ctx.features[i] = cvPoint2D32f(-1, -1);
		ctx.feature_failed_iters = new char[sz];
		for (int i = 0; i < sz; i++)
			ctx.feature_failed_iters[i] = 0;
	}

	for (int i = 0; i < ctx.model.count; i++) {
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

	vector<KeyPoint> corners;

	Mat mat = Mat(Mat(ctx.grayscale, false), roi);
	int max = ctx.state == HT_STATE_TRACKING
		? ctx.config.max_tracked_features * ctx.config.max_detect_features
		: ctx.config.min_track_start_features * 1.5f;

	int good = 0;

redetect:
	ORB detector = ORB(max*20, 1.1f, 12, ctx.config.feature_quality_level, 0, 2, 0, ctx.config.feature_quality_level);
	detector(mat, noArray(), corners);

	printf("ORB gave %d corners at quality level %d\n", corners.size(), ctx.config.feature_quality_level);

	sort(corners.begin(), corners.end(), ht_feature_quality_level);

	int count = corners.size(), k = 0;

	if (count == 0)
		return;

	CvPoint2D32f* features_to_add = new CvPoint2D32f[count];

	for (int i = 0; i < count; i++) {
		corners[i].pt.x += roi.x;
		corners[i].pt.y += roi.y;
		
		triangle_t t;
		int idx;

		if (!(ht_triangle_at(ctx, corners[i].pt, &t, &idx, rotation_matrix, translation_vector, model)))
			continue;

		good++;

		if (ctx.features[idx].x != -1)
			continue;

		features_to_add[k++] = corners[i].pt;
	}

	if (good > max) {
		if (ctx.config.feature_quality_level < HT_FEATURE_MAX_QUALITY_LEVEL)
			ctx.config.feature_quality_level++;
	} else {
		if (ctx.config.feature_quality_level > HT_FEATURE_MIN_QUALITY_LEVEL) {
			ctx.config.feature_quality_level--;
			if (ctx.state == HT_STATE_INITIALIZING) {
				corners.clear();
				delete[] features_to_add;
				goto redetect;
			}
		}
	}

	if (k > max)
		k = max;

	float max_distance = ctx.config.min_feature_distance * ctx.zoom_ratio * ctx.config.min_feature_distance * ctx.zoom_ratio;

	if (k > 0) {
		if (roi.width > 17 && roi.height > 13)
			cvFindCornerSubPix(ctx.grayscale, features_to_add, k, cvSize(8, 6), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 0.05));
		for (int i = 0; i < k && ctx.feature_count < ctx.config.max_tracked_features; i++) {
			triangle_t t;
			int idx;

			if (!(ht_triangle_at(ctx, features_to_add[i], &t, &idx, rotation_matrix, translation_vector, model)))
				continue;

			if (ctx.features[idx].x != -1 || ctx.features[idx].y != -1)
				continue;

			for (int j = 0; j < model.count; j++)
				if (ctx.features[j].x != -1 && ht_distance2d_squared(features_to_add[i], ctx.features[j]) < max_distance)
					goto end2;

			ctx.features[idx] = features_to_add[i];
			ctx.feature_count++;
	end2:
			;
		}
	}

	delete[] features_to_add;
}
