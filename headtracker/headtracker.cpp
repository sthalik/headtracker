#include "stdafx.h"

using namespace std;
using namespace cv;

static void ht_mouse_callback(int, int x, int y, int, void* param) {
	headtracker_t& ctx = *(headtracker_t*) param;

	ctx.mouse_x = x;
	ctx.mouse_y = y;
}

int _tmain(int argc, _TCHAR* argv[])
{
	float rotation_matrix[9];
	float translation_vector[3];
	srand((int) getTickCount());
	headtracker_t* ctx = ht_make_context(0);

	cvNamedWindow("capture", CV_WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
	cvSetMouseCallback("capture", ht_mouse_callback, ctx);

	int mx = -1, my = -1;

	while (1) {
		if (!ht_get_image(*ctx))
			break;

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
						printf("retries: %d; feature count=%d\n", ctx->init_retries, ctx->feature_count);
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
				if (ht_ransac_best_indices(*ctx, &best_cnt, &best_error, best_indices) &&
					ht_estimate_pose(*ctx, rotation_matrix, translation_vector, best_indices, best_cnt, &offset))
				{
					ht_remove_lumps(*ctx);
					ht_project_model(*ctx, rotation_matrix, translation_vector, ctx->model, cvPoint3D32f(-offset.x, -offset.y, -offset.z));
					ht_get_features(*ctx, rotation_matrix, translation_vector, ctx->model);
					ht_draw_model(*ctx, rotation_matrix, translation_vector, ctx->model);
					ht_draw_features(*ctx);
				} else
					ctx->state = HT_STATE_LOST;
				delete[] best_indices;
#if 0
				if (ctx->mouse_x != mx || ctx->mouse_y != my) {
						mx = ctx->mouse_x;
						my = ctx->mouse_y;
						triangle_t t;
						int idx;
						if (ht_triangle_at(*ctx, cvPoint(ctx->mouse_x, ctx->mouse_y), &t, &idx, rotation_matrix, translation_vector, ctx->model)) {
							printf("MOUSE: %d %f %f %f\n", idx+1, (t.p1.x + t.p2.x + t.p3.x) / 3, (t.p1.y + t.p2.y + t.p3.y) / 3, (t.p1.z + t.p2.z + t.p3.z) / 3);
						} else {
							printf("MOUSE: no triangle\n");
						}
					}
#endif
				if (ctx->state == HT_STATE_TRACKING) {
					printf("%.2f %.2f %d\n",
						   best_error.avg,
						   // this happens because of remove_lumps() and friends
						   // don't be alarmed
						   max(0.0f, 1.0f - best_cnt / (float) ctx->feature_count),
						   ctx->feature_count);
				}
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
			default: {
				throw exception("unknown state");
			}
		}

		cvShowImage("capture", ctx->color);
		cvWaitKey(1);

		if (!ctx->last_image)
			ctx->last_image = cvCreateImage(cvGetSize(ctx->grayscale), IPL_DEPTH_8U, 1);
		cvCopy(ctx->grayscale, ctx->last_image);
	}

	ht_free_context(ctx);

	return 0;
}