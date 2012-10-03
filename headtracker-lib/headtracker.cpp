#include "stdafx.h"

using namespace std;
using namespace cv;

#include <string>

HT_API(void) ht_reset(headtracker_t* ctx) {
    ctx->state = HT_STATE_LOST;
}

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
        ctx->start_frames = ctx->config.feature_good_nframes;
		if (!(ctx->focal_length > 0)) {
            float inv_ar = (ctx->grayscale.rows / (float) ctx->grayscale.cols);
            ctx->focal_length_w = ctx->grayscale.cols / tan(0.5 * ctx->config.field_of_view * HT_PI / 180.0);
            ctx->focal_length_h = ctx->grayscale.rows / tan(0.5 * ctx->config.field_of_view * inv_ar * HT_PI / 180.0);
            ctx->focal_length = (ctx->focal_length_w + ctx->focal_length_h) * 0.5;
            fprintf(stderr, "focal length = %f\n", ctx->focal_length);
		}
        if (ht_initial_guess(*ctx, ctx->grayscale, rotation_matrix, translation_vector))
		{
			ht_project_model(*ctx, rotation_matrix, translation_vector, ctx->model, cvPoint3D32f(0, 0, 0));
            imshow("grayscale", ctx->grayscale);
            ht_track_features(*ctx);
            ht_get_features(*ctx, ctx->model);
            if (ctx->keypoint_count >= ctx->config.max_keypoints * 2 / 3) {
                float best_error;
                if (ht_ransac_best_indices(*ctx, &best_error))
                {
                    ctx->state = HT_STATE_TRACKING;
                    ctx->restarted = false;
                } else {
                    if (++ctx->init_retries > ctx->config.max_init_retries)
                        ctx->state = HT_STATE_LOST;
                }
            }
		}
		break;
    } case HT_STATE_TRACKING: {
        ctx->start_frames = max(0, ctx->start_frames - 1);
        imshow("grayscale", ctx->grayscale);
        ht_track_features(*ctx);
        float best_error = 1.0e10;
        CvPoint3D32f offset;

        if (ht_ransac_best_indices(*ctx, &best_error) &&
            ht_estimate_pose(*ctx, rotation_matrix, translation_vector, rotation_matrix2, translation_vector2, &offset))
        {
            ht_project_model(*ctx, rotation_matrix, translation_vector, ctx->model, cvPoint3D32f(offset.x, offset.y, offset.z));
            ht_draw_model(*ctx, ctx->model);
            ht_draw_features(*ctx);
            ctx->hz++;
            int ticks = ht_tickcount() / 1000;
            if (ctx->ticks_last_second != ticks) {
                ctx->ticks_last_second = ticks;
                ctx->hz_last_second = ctx->hz;
                ctx->hz = 0;
                fprintf(stderr, "hz: %d\n", ctx->hz_last_second);
            }
            if (ctx->hz_last_second != -1) {
                char buf2[42];
                string buf;
                buf.append("Hz: ");
                sprintf(buf2, "%d", ctx->hz_last_second);
                buf.append(buf2);
                putText(ctx->color, buf, Point(30, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
            }
            ht_remove_outliers(*ctx);
            ht_get_features(*ctx, ctx->model);
            *euler = ht_matrix_to_euler(rotation_matrix2, translation_vector2);
			euler->filled = true;
            euler->confidence = -best_error;
			if (ctx->config.debug)
                printf("keypoints %d/%d (%d); confidence=%f; dist=%f\n",
                       ctx->keypoint_count,
                       ctx->config.max_keypoints,
                       ctx->config.keypoint_quality,
                       -best_error,
                       ctx->zoom_ratio);
        } else {
            if (ctx->abortp)
                abort();
			ctx->state = HT_STATE_LOST;
        }
		break;
	} case HT_STATE_LOST: {
		ctx->state = HT_STATE_INITIALIZING;
		ctx->init_retries = 0;
		ctx->restarted = true;
		ctx->depth_counter_pos = 0;
		ctx->zoom_ratio = 1.0f;
		ctx->keypoint_count = 0;
        ctx->abortp = false;
		for (int i = 0; i < ctx->config.max_keypoints; i++)
			ctx->keypoints[i].idx = -1;
        ctx->hz = 0;
		break;
	}
	default:
		return false;
	}

    ctx->grayscale.copyTo(ctx->last_image);

	return true;
}
