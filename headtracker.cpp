#undef NDEBUG
#include <cassert>
#include "ht-internal.h"
#include "sleep.hpp"
#include <string>

#define SSTR( x ) ((std::ostringstream &) ( \
        ( std::ostringstream() << std::dec << x ) )).str()

context::context(const ht_config& conf) :
    config(conf), camera(conf.camera_index), model("head.raw"), bbox("bounding-box.raw"),
    state(STATE_LOST), hz(0), iter_time(0), flandmark_model(nullptr)
{
    if (conf.force_width)
        camera.set(CV_CAP_PROP_FRAME_WIDTH, conf.force_width);
    if (conf.force_height)
        camera.set(CV_CAP_PROP_FRAME_HEIGHT, conf.force_height);
    if (conf.force_fps)
        camera.set(CV_CAP_PROP_FPS, conf.force_fps);
    
    flandmark_model = flandmark_init("flandmark_model.dat");
    
    if (!flandmark_model)
        assert(!"flandmark model missing/corrupt");
    
    if (!camera.isOpened())
        assert(!"can't open camera");
    
    int w, h;
    
    {
        bool ok = false;
        for (int i = 0; i < 100; i++)
        {
            cv::Mat frame;
            if (!camera.retrieve(frame))
            {
                portable::sleep(5);
                continue;
            }
            ok = true;
            w = frame.cols;
            h = frame.rows;
            break;
        }
        
        if (!ok)
            assert(!"can't open camera");
    }
    
    {
        const double diag = sqrt(w * w + h * h)/w, diag_fov = conf.field_of_view;
        const double fov_w = 2.*atan(tan(diag_fov/2.)/sqrt(1. + h/(double)w * h/(double)w));
        const double fov_h = 2.*atan(tan(diag_fov/2.)/sqrt(1. + w/(double)h * w/(double)h));
        
        intrins = cv::Matx33f::eye();
        intrins(0, 0) = .5 * w / tan(.5 * fov_w); // fx
        intrins(1, 1) = .5 * h / tan(.5 * fov_h); // fy
        intrins(0, 2) = w/2.;
        intrins(1, 2) = h/2.;
    }
}

ht_result context::emit_result(const cv::Matx31d& rvec, const cv::Matx31d& tvec)
{
    ht_result ret;
    
    cv::Matx33d m_R, m_Q, rmat;
    cv::Rodrigues(rvec, rmat);
    cv::Vec3d foo = cv::RQDecomp3x3(rmat, m_R, m_Q);
    
    ret.rotx = foo[1];
    ret.roty = foo[0];
    ret.rotz = foo[2];
    ret.tx = tvec(0, 0);
    ret.ty = tvec(0, 1);
    ret.tz = tvec(0, 2);
	ret.filled = true;

    return ret;
}

#if 0
HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler) {
    euler->filled = false;

	if (!ht_get_image(*ctx))
		return false;

    switch (ctx->state) {
	case HT_STATE_INITIALIZING: {
        Mat rvec, tvec;
        if (ht_initial_guess(*ctx, ctx->grayscale, rvec, tvec) &&
            ht_project_model(*ctx, rvec, tvec, ctx->model) &&
            ht_project_model(*ctx, rvec, tvec, ctx->bbox))
		{
            //ht_draw_model(*ctx, ctx->model);
			ctx->zoom_ratio = fabs(ctx->focal_length_w * 0.25 / tvec.at<double>(2));
			Rect roi = ht_get_bounds(*ctx, ctx->bbox);
			if (roi.width > 5 && roi.height > 5)
			{
				ht_track_features(*ctx);
				ht_get_features(*ctx, ctx->model);
				ctx->state = HT_STATE_TRACKING;
				ctx->restarted = false;
			}
		}
		break;
    } case HT_STATE_TRACKING: {
#if 0
            //if (ctx->config.debug)
            {
                imshow("bw", ctx->grayscale);
                waitKey(1);
            }
#endif
            float error = 0;
            Mat rvec, tvec;
            Rect roi;
            
			ht_track_features(*ctx);

            if (ht_ransac_best_indices(*ctx, error, rvec, tvec) &&
                (ctx->zoom_ratio = ctx->focal_length_w * 0.25 / tvec.at<double>(2)) > 0 &&
                error < ctx->config.ransac_max_mean_error * ctx->zoom_ratio &&
                error < ctx->config.ransac_abs_max_mean_error &&
                ht_project_model(*ctx, rvec, tvec, ctx->model) &&
                ht_project_model(*ctx, rvec, tvec, ctx->bbox) &&
                ((roi = ht_get_bounds(*ctx, ctx->bbox)), (roi.width > 5 && roi.height > 5)))
            {
                ht_draw_model(*ctx, ctx->model);
                if (ctx->config.debug)
                {
                    ht_draw_features(*ctx);
                    Scalar color(0, 0, 255);
                    rectangle(ctx->color, roi, color, 2);
                }
                ctx->hz++;
                int ticks = ht_tickcount() / 1000;
                if (ctx->ticks_last_second != ticks) {
                    ctx->ticks_last_second = ticks;
                    ctx->hz_last_second = ctx->hz;
                    ctx->hz = 0;
                }
                if (ctx->hz_last_second != -1) {
                    const double scale = ctx->grayscale.cols > 480 ? 1 : 0.5;
                    string buf;
                    buf.append("Hz: ");
                    buf.append(SSTR(ctx->hz_last_second));
                    putText(ctx->color, buf, Point(10, 30), FONT_HERSHEY_PLAIN, scale * 2.56, Scalar(0, 255, 0), 2);
					buf.clear();
					buf.append("Error: ");
					buf.append(SSTR(error));
					putText(ctx->color, buf, Point(10, 60), FONT_HERSHEY_PLAIN, scale * 2.56, Scalar(0, 255, 0), 2);
					buf.clear();
                    buf.append("Keypoints: ");
					int cnt = 0;
					for (int i = 0; i < ctx->config.max_keypoints; i++)
						if (ctx->keypoints[i].idx != -1)
							cnt++;
					buf.append(SSTR(cnt));
					putText(ctx->color, buf, Point(10, 90), FONT_HERSHEY_PLAIN, scale * 2.56, Scalar(0, 255, 0), 2);
                }
				ctx->has_pose = true;
				ctx->rvec = rvec.clone();
				ctx->tvec = tvec.clone();
                ht_get_next_features(*ctx, roi);
                *euler = ht_matrix_to_euler(rvec, tvec);
                euler->filled = true;
                //euler->rotx -= atan(euler->tx / euler->tz) * 180 / HT_PI;
                //euler->roty += atan(euler->ty / euler->tz) * 180 / HT_PI;
        } else {
			if (ctx->config.debug)
				fprintf(stderr, "bad roi %d %d; err=%f\n", roi.width, roi.height, error);
			ctx->state = HT_STATE_LOST;
		}
		break;
	} case HT_STATE_LOST: {
		ctx->state = HT_STATE_INITIALIZING;
		ctx->restarted = true;
        ctx->zoom_ratio = 1;
        for (int i = 0; i < ctx->config.max_keypoints; i++)
			ctx->keypoints[i].idx = -1;
        ctx->hz = 0;
		ctx->has_pose = false;
		break;
	}

	default:
		return false;
	}

	return true;
}
#endif