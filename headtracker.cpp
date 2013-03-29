#include "ht-api.h"
#include "ht-internal.h"
using namespace std;
using namespace cv;

#include <string>

#define SSTR( x ) ((std::ostringstream &) ( \
        ( std::ostringstream() << std::dec << x ) )).str()

HT_API(void) ht_reset(headtracker_t* ctx) {
    ctx->state = HT_STATE_LOST;
}

Rect ht_get_roi(headtracker_t &ctx, model_t &model) {
	Rect rect(65535, 65535, 0, 0);
	Rect rect2(65535, 65535, 0, 0);

	if (ctx.has_pose)
	{
		vector<Point3f> points1(4), points3(4);
		vector<Point2f> points2, points4;

		points1[0] = Point3f(-10, -9, 12);
		points1[1] = Point3f(10, -9, 12);
		points1[2] = Point3f(-4, 9, 10);
		points1[3] = Point3f(4, 9, 10);

		points3[0] = Point3f(-17, -15, 12);
		points3[1] = Point3f(17, -15, 12);
		points3[2] = Point3f(-9, 13, 10);
		points3[3] = Point3f(9, 13, 10);

	    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
		intrinsics.at<float> (0, 0) = ctx.focal_length_w;
		intrinsics.at<float> (1, 1) = ctx.focal_length_h;
		intrinsics.at<float> (0, 2) = ctx.grayscale.cols/2;
		intrinsics.at<float> (1, 2) = ctx.grayscale.rows/2;

		Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);

		projectPoints(points1, ctx.rvec, ctx.tvec, intrinsics, dist_coeffs, points2);
		projectPoints(points3, ctx.rvec, ctx.tvec, intrinsics, dist_coeffs, points4);

		for (int i = 0; i < points2.size(); i++)
		{
			rect.x = min<int>(points2[i].x, rect.x);
			rect.y = min<int>(points2[i].y, rect.y);
			rect2.x = min<int>(points4[i].x, rect2.x);
			rect2.y = min<int>(points4[i].y, rect2.y);
		}

		for (int i = 0; i < points2.size(); i++)
		{
			rect.width = max<int>(points2[i].x - rect.x, rect.width);
			rect.height = max<int>(points2[i].y - rect.y, rect.height);
			rect2.width = max<int>(points4[i].x - rect2.x, rect2.width);
			rect2.height = max<int>(points4[i].y - rect2.y, rect2.height);
		}

		if (ctx.config.debug)
		{
			printf("rect = (%d, %d, %d, %d)\n", rect.x, rect.y, rect.width, rect.height);
			printf("rect2 = (%d, %d, %d, %d)\n", rect2.x, rect2.y, rect2.width, rect2.height);
		}
	}
	else {
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

		rect = Rect(min_x-width*50/100, min_y-height*30/100, width*200/100, height*136/100);
		rect2 = Rect(min_x-width*90/100, min_y-height*8/10, width*280/100, height*24/10);

		if (ctx.config.debug)
		{
			printf("rect = (%d, %d, %d, %d)\n", rect.x, rect.y, rect.width, rect.height);
			printf("rect2 = (%d, %d, %d, %d)\n", rect2.x, rect2.y, rect2.width, rect2.height);
		}
	}

    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    if (rect.width + rect.x > ctx.grayscale.cols)
        rect.width = ctx.grayscale.cols - rect.x;
    if (rect.height + rect.y > ctx.grayscale.rows)
        rect.height = ctx.grayscale.rows - rect.y;
    if (ctx.config.debug)
    {
        Scalar color(255, 0, 0);
        rectangle(ctx.color, rect, color, 2);
	}

    if (rect2.x < 0)
        rect2.x = 0;
    if (rect2.y < 0)
        rect2.y = 0;
    if (rect2.width + rect2.x > ctx.grayscale.cols)
        rect2.width = ctx.grayscale.cols - rect2.x;
    if (rect2.height + rect2.y > ctx.grayscale.rows)
        rect2.height = ctx.grayscale.rows - rect2.y;

    if (ctx.config.debug)
    {
        Scalar color(0, 255, 0);
        rectangle(ctx.color, rect2, color, 2);
    }

    Mat foo = ctx.tmp(rect2);
    equalizeHist(foo, ctx.grayscale(rect2));

    return rect;
}

static ht_result_t ht_matrix_to_euler(const Mat& rvec, const Mat& tvec) {
    ht_result_t ret;
    Mat rotation_matrix = Mat::zeros(3, 3, CV_64FC1);
    
    Mat junk1(3, 3, CV_64FC1), junk2(3, 3, CV_64FC1);

    Rodrigues(rvec, rotation_matrix);

    Vec3d foo = cv::RQDecomp3x3(rotation_matrix, junk1, junk2);
    
    ret.rotx = foo[1];
    ret.roty = foo[0];
    ret.rotz = foo[2];
    ret.tx = tvec.at<double>(0, 0);
    ret.ty = tvec.at<double>(0, 1);
    ret.tz = tvec.at<double>(0, 2);

    return ret;
}

static void ht_get_next_features(headtracker_t& ctx, const Rect roi)
{
    if (ctx.state = HT_STATE_TRACKING) {
        int val = ctx.dropped++;
        ctx.dropped %= 3;
        if (val != 0)
            return;
    }
    Mat rvec, tvec;
    //if (!ht_initial_guess(ctx, ctx.tmp, rvec, tvec))
    if (!ht_fl_estimate(ctx, ctx.tmp, roi, rvec, tvec))
        return;
    model_t tmp_model;

    tmp_model.triangles = ctx.model.triangles;
    tmp_model.count = ctx.model.count;
    tmp_model.projection = new triangle2d_t[ctx.model.count];
    tmp_model.rotation = new triangle_t[ctx.model.count];

    ht_project_model(ctx, rvec, tvec, tmp_model);
    ht_get_features(ctx, tmp_model);
    delete[] tmp_model.projection;
    delete[] tmp_model.rotation;
}

HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler) {
    euler->filled = false;

	if (!ht_get_image(*ctx))
		return false;

    switch (ctx->state) {
	case HT_STATE_INITIALIZING: {
        if (!(ctx->focal_length_w > 0)) {
            ctx->focal_length_w = -ctx->grayscale.cols / tan(ctx->config.field_of_view * HT_PI / 180);
            //ctx->focal_length_h = ctx->focal_length_w;
            ctx->focal_length_h = -ctx->grayscale.rows / tan(ctx->config.field_of_view
                * ctx->grayscale.rows / ctx->grayscale.cols
                * HT_PI / 180.0);
            if (ctx->config.debug)
                fprintf(stderr, "focal length = %f\n", ctx->focal_length_w);
        }
        Mat rvec, tvec;
        if (ht_initial_guess(*ctx, ctx->grayscale, rvec, tvec))
		{
            ht_project_model(*ctx, rvec, tvec, ctx->model);
            ht_draw_model(*ctx, ctx->model);
            Rect roi = ht_get_roi(*ctx, ctx->model);
            float error = 1e4;
            if (roi.width > 5 && roi.height > 5)
            {
                ht_track_features(*ctx);
                ht_get_features(*ctx, ctx->model);
                if (ctx->config.debug)
                    ht_draw_features(*ctx);
                if (ht_ransac_best_indices(*ctx, error, rvec, tvec)) {
                    ctx->restarted = false;
                    ctx->state = HT_STATE_TRACKING;
                }
            }
		}
		break;
    } case HT_STATE_TRACKING: {
        Rect roi = ht_get_roi(*ctx, ctx->model);
        if (roi.width > 5 && roi.height > 5)
        {
#if 0
            if (ctx->config.debug)
            {
                imshow("bw", ctx->grayscale);
                waitKey(1);
            }
#endif
            ht_track_features(*ctx);
            float error = 0;
            Mat rvec, tvec;

            if (ht_ransac_best_indices(*ctx, error, rvec, tvec) &&
                error < ctx->config.ransac_max_mean_error * ctx->zoom_ratio &&
                error < ctx->config.ransac_abs_max_mean_error)
            {
                ctx->zoom_ratio = ctx->focal_length_w * 0.15 / tvec.at<double>(2);
                if (ctx->config.debug) {
    				printf("zoom_ratio = %f\n", ctx->zoom_ratio);
                }
                ht_project_model(*ctx, rvec, tvec, ctx->model);
                ht_draw_model(*ctx, ctx->model);
                if (ctx->config.debug)
                    ht_draw_features(*ctx);
                ctx->hz++;
                int ticks = ht_tickcount() / 1000;
                if (ctx->ticks_last_second != ticks) {
                    ctx->ticks_last_second = ticks;
                    ctx->hz_last_second = ctx->hz;
                    ctx->hz = 0;
                }
                if (ctx->hz_last_second != -1) {
                    string buf;
                    buf.append("Hz: ");
                    buf.append(SSTR(ctx->hz_last_second));
                    putText(ctx->color, buf, Point(10, 40), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2);
                }
                ht_get_next_features(*ctx, roi);
                *euler = ht_matrix_to_euler(rvec, tvec);
                euler->filled = true;
                ctx->has_pose = true;
				ctx->rvec = rvec.clone();
                ctx->tvec = tvec.clone();
            } else {
				if (ctx->config.debug)
					fprintf(stderr, "error %f\n", error);
                ctx->state = HT_STATE_LOST;
            }
        } else {
			if (ctx->config.debug)
				fprintf(stderr, "bad roi\n");
			ctx->state = HT_STATE_LOST;
		}
		break;
	} case HT_STATE_LOST: {
		ctx->state = HT_STATE_INITIALIZING;
		ctx->restarted = true;
        ctx->zoom_ratio = 1;
        for (int i = 0; i < ctx->config.max_keypoints; i++)
			ctx->keypoints[i].idx = -1;
        ctx->has_pose = false;
        ctx->hz = 0;
        ctx->dropped = 0;
		break;
	}

	default:
		return false;
	}

	return true;
}
