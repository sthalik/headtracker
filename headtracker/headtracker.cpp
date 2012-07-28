#include "stdafx.h"

using namespace std;
using namespace cv;

HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler) {
	float rotation_matrix[9];
	float translation_vector[3];
	float rotation_matrix2[9];
	float translation_vector2[3];

	euler->filled = false;

	if (!ht_get_image(*ctx))
		return false;
		
	switch (ctx->state) {
	case HT_STATE_INITIALIZING: {
		ht_track_features(*ctx);
		if (ht_initial_guess(*ctx, *ctx->grayscale, rotation_matrix, translation_vector)) {
			ht_project_model(*ctx, rotation_matrix, translation_vector, ctx->model, cvPoint3D32f(0, 0, 0));
			ht_get_features(*ctx, rotation_matrix, translation_vector, ctx->model);
			ctx->restarted = false;
			error_t best_error;
			int best_cnt;
			int* best_indices = new int[ctx->feature_count];
			if (ht_ransac_best_indices(*ctx, &best_cnt, &best_error, best_indices) && ctx->feature_count >= ctx->config.min_track_start_features)
				ctx->state = HT_STATE_TRACKING;
			else {
				if (++ctx->init_retries > ctx->config.max_init_retries)
					ctx->state = HT_STATE_LOST;
			}
			delete[] best_indices;
		}
		break;
	} case HT_STATE_TRACKING: {
		ht_track_features(*ctx);
		error_t best_error;
		int best_cnt;
		int* best_indices = new int[ctx->feature_count];
		CvPoint3D32f offset;
		CvPoint2D32f centroid;
		if (ht_ransac_best_indices(*ctx, &best_cnt, &best_error, best_indices) &&
			ht_estimate_pose(*ctx, rotation_matrix, translation_vector, rotation_matrix2, translation_vector2, best_indices, best_cnt, &offset, &centroid))
		{
			ht_remove_lumps(*ctx);
			ht_project_model(*ctx, rotation_matrix, translation_vector, ctx->model, cvPoint3D32f(offset.x, offset.y, offset.z));
			ht_get_features(*ctx, rotation_matrix, translation_vector, ctx->model);
			ht_draw_model(*ctx, rotation_matrix, translation_vector, ctx->model);
			ht_draw_features(*ctx);
			*euler = ht_matrix_to_euler(rotation_matrix2, translation_vector2);
			euler->filled = true;
			euler->confidence = (ctx->config.ransac_max_consensus_error - best_error.avg) / ctx->config.ransac_max_consensus_error;
			euler->feature_ratio = best_cnt / (float) ctx->config.max_tracked_features;
			cvCircle(ctx->color, cvPoint(centroid.x, centroid.y), 3, CV_RGB(0, 255, 0), -1);
		} else
			ctx->state = HT_STATE_LOST;
		delete[] best_indices;
		break;
	} case HT_STATE_LOST: {
		ctx->feature_count = 0;
		for (int i = 0; i < ctx->model.count; i++) {
			ctx->features[i] = cvPoint2D32f(-1, -1);
			ctx->feature_failed_iters[i] = 0;
		}
		ctx->state = HT_STATE_INITIALIZING;
		ctx->init_retries = 0;
		ctx->restarted = true;
		ctx->depth_frame_count = 0;
		ctx->depth_counter_pos = 0;
		ctx->zoom_ratio = 1.0f;
		break;
	}
	default:
		throw exception();
	}

	if (!ctx->last_image)
		ctx->last_image = cvCreateImage(cvGetSize(ctx->grayscale), IPL_DEPTH_8U, 1);
	cvCopy(ctx->grayscale, ctx->last_image);

	if (!ctx->bgr_frame)
		ctx->bgr_frame = new char[ctx->color->width * ctx->color->height * 3];

	memcpy(ctx->bgr_frame, ctx->color->imageData, ctx->color->width * ctx->color->height * 3);

	return true;
}