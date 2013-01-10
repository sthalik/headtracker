#include "stdafx.h"

using namespace std;
using namespace cv;

#include <string>

HT_API(void) ht_reset(headtracker_t* ctx) {
    ctx->state = HT_STATE_LOST;
}

Rect ht_get_roi(const headtracker_t &ctx, model_t &model) {
    float min_x = (float) ctx.grayscale.cols, max_x = 0.0f;
    float min_y = (float) ctx.grayscale.rows, max_y = 0.0f;

    for (int i = 0; i < ctx.model.count; i++) {
        float minx = min(model.projection[i].p1.x, min(model.projection[i].p2.x, model.projection[i].p3.x));
        float maxx = max(model.projection[i].p1.x, max(model.projection[i].p2.x, model.projection[i].p3.x));
        float miny = min(model.projection[i].p1.y, min(model.projection[i].p2.y, model.projection[i].p3.y));
        float maxy = max(model.projection[i].p1.y, max(model.projection[i].p2.y, model.projection[i].p3.y));
        if (maxx > max_x)
            max_x = maxx;
        if (minx < min_x)
            min_x = minx;
        if (maxy > max_y)
            max_y = maxy;
        if (miny < min_y)
            min_y = miny;
    }

    int width = max_x - min_x;
    int height = max_y - min_y;

    Rect rect = Rect(min_x-width/3, min_y-height/3, width*5/3, height*5/3);

    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    if (rect.width + rect.x > ctx.grayscale.cols)
        rect.width = ctx.grayscale.cols - rect.x;
    if (rect.height + rect.y > ctx.grayscale.rows)
        rect.height = ctx.grayscale.rows - rect.y;

    return rect;
}

static void ht_get_face_histogram(headtracker_t& ctx, const Rect roi) {
    equalizeHist(ctx.tmp(roi), ctx.face_histogram);
    ctx.face_histogram.copyTo(ctx.grayscale(roi));
}

static ht_result_t ht_matrix_to_euler(const Mat& rvec, const Mat& tvec) {
    ht_result_t ret;
    Mat rotation_matrix = Mat::zeros(3, 3, CV_64FC1);

    Rodrigues(rvec, rotation_matrix);

    if (rotation_matrix.at<double>(0, 2) > 0.9998) {
        ret.rotx = HT_PI/2.0;
        ret.roty = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1));
        ret.rotz = 0.0f;
    } else if (rotation_matrix.at<double>(0, 2) < -0.9998) {
        ret.rotx = HT_PI/-2.0;
        ret.roty = -atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1));
        ret.rotz = 0.0f;
    } else {
        ret.rotx = asin(rotation_matrix.at<double>(0, 2));
        ret.roty = -atan2(rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(2, 2));
        ret.rotz = atan2(-rotation_matrix.at<double>(0, 1), rotation_matrix.at<double>(0 ,0));
    }

    ret.tx = tvec.at<double>(0, 0) / 10;
    ret.ty = tvec.at<double>(0, 1) / 10;
    ret.tz = tvec.at<double>(0, 2) / 10;

    return ret;
}

static void ht_get_next_features(headtracker_t& ctx, const Rect roi)
{
    Mat rvec, tvec;
    if (!ht_fl_estimate(ctx, ctx.grayscale, roi, rvec, tvec))
    //if (!ht_initial_guess(ctx, ctx.tmp, rvec, tvec))
    {
        if (ctx.config.debug)
            printf("Can't locate face!\n");
        return;
    }
    else
    {
        if (ctx.config.debug)
            printf("Face located!\n");
    }
    model_t tmp_model;

    tmp_model.triangles = ctx.model.triangles;
    tmp_model.count = ctx.model.count;
    tmp_model.projection = new triangle2d_t[ctx.model.count];

    ht_project_model(ctx, rvec, tvec, tmp_model);
    ht_get_features(ctx, tmp_model);
    delete[] tmp_model.projection;
}

HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler) {
    euler->filled = false;

	if (!ht_get_image(*ctx))
		return false;

	switch (ctx->state) {
	case HT_STATE_INITIALIZING: {
        if (!(ctx->focal_length_w > 0)) {
            ctx->focal_length_w = ctx->grayscale.cols / tan(0.5 * ctx->config.field_of_view * HT_PI / 180.0);
            ctx->focal_length_h = ctx->focal_length_w;
            //ctx->focal_length_h = ctx->grayscale.rows / tan(0.5 * ctx->config.field_of_view * (ctx->grayscale.rows / (float) ctx->grayscale.cols) * HT_PI / 180.0);
            fprintf(stderr, "focal length = %f\n", ctx->focal_length_w);
        }
        ht_draw_features(*ctx);
        Mat rvec, tvec;
        if (ht_initial_guess(*ctx, ctx->grayscale, rvec, tvec))
		{
            ht_project_model(*ctx, rvec, tvec, ctx->model);
            Rect roi = ht_get_roi(*ctx, ctx->model);
            ht_get_face_histogram(*ctx, roi);
            ht_track_features(*ctx);
            ht_get_features(*ctx, ctx->model);
            ctx->restarted = false;
            float error = 0;
            if (ctx->config.debug)
                printf("INIT: got %d/%d keypoints (%d)\n",
                       ctx->keypoint_count,
                       ctx->config.max_keypoints,
                       ctx->config.keypoint_quality);
            if (ctx->keypoint_count >= 4) {
                if (ht_ransac_best_indices(*ctx, error, rvec, tvec))
                {
                    ctx->rvec = rvec;
                    ctx->tvec = tvec;
                    ctx->state = HT_STATE_TRACKING;
                }
            }
		}
		break;
    } case HT_STATE_TRACKING: {
        Rect roi = ht_get_roi(*ctx, ctx->model);
        ht_get_face_histogram(*ctx, roi);
        //imshow("bw", ctx->grayscale);
        ht_track_features(*ctx);
        float error = 0;
        Mat rvec, tvec;

        if (ctx->has_pose) {
            rvec = ctx->rvec;
            tvec = ctx->tvec;
        }

        if (ht_ransac_best_indices(*ctx, error, rvec, tvec) &&
            error < ctx->config.ransac_max_mean_error * ctx->zoom_ratio &&
            error < ctx->config.ransac_abs_max_mean_error)
        {
            ctx->rvec = rvec;
            ctx->tvec = tvec;
            ctx->zoom_ratio = HT_STD_DEPTH / tvec.at<double>(2);
            ht_project_model(*ctx, rvec, tvec, ctx->model);
            ht_draw_model(*ctx, ctx->model);
            ht_draw_features(*ctx);
            ctx->hz++;
            int ticks = ht_tickcount() / 1000;
            if (ctx->ticks_last_second != ticks) {
                ctx->ticks_last_second = ticks;
                ctx->hz_last_second = ctx->hz;
                ctx->hz = 0;
            }
            if (ctx->hz_last_second != -1) {
                char buf2[42];
                string buf;
                buf.append("Hz: ");
                sprintf(buf2, "%d", ctx->hz_last_second);
                buf.append(buf2);
                putText(ctx->color, buf, Point(30, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));
            }
            //ht_remove_outliers(*ctx);
            ht_get_next_features(*ctx, roi);
            *euler = ht_matrix_to_euler(rvec, tvec);
			euler->filled = true;
            if (ctx->config.debug)
            {
                printf("keypoints %d/%d (%d); dist=%f; error=%f\n",
                       ctx->keypoint_count,
                       ctx->config.max_keypoints,
                       ctx->config.keypoint_quality,
                       ctx->zoom_ratio,
                       error);
            }
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
		ctx->zoom_ratio = 1.0f;
        ctx->keypoint_count = 0;
        ctx->abortp = false;
        for (int i = 0; i < ctx->config.max_keypoints; i++)
			ctx->keypoints[i].idx = -1;
        ctx->has_pose = false;
        ctx->hz = 0;
		break;
	}
	default:
		return false;
	}

    ctx->grayscale.copyTo(ctx->last_image);

	return true;
}
