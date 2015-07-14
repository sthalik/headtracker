#include "ht-api.h"
#include "ht-internal.h"
using namespace std;
using namespace cv;

bool ht_ransac_best_indices(headtracker_t& ctx, float& mean_error, Mat& rvec_, Mat& tvec_) {
    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = ctx.focal_length_w;
    intrinsics.at<float> (1, 1) = ctx.focal_length_h;
    intrinsics.at<float> (0, 2) = ctx.grayscale.cols/2;
    intrinsics.at<float> (1, 2) = ctx.grayscale.rows/2;

    Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);
    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);
    
    for (int i = 0; i < 5; i++)
        dist_coeffs.at<float>(i) = ctx.config.dist_coeffs[i];

    rvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (0, 0) = 1.0;
    tvec.at<double> (1, 0) = 1.0;

    vector<Point3f> object_points;
    vector<Point2f> image_points;
    
    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx == -1)
            continue;
        object_points.push_back(ctx.keypoint_uv[i]);
        image_points.push_back(ctx.keypoints[i].position);
    }

    if (object_points.size() >= 15)
    {
		if (ctx.has_pose)
		{
			rvec = ctx.rvec.clone();
			tvec = ctx.tvec.clone();
		}
        if (!solvePnPRansac(object_points,
                            image_points,
                            intrinsics,
                            dist_coeffs,
                            rvec,
                            tvec,
                            ctx.has_pose,
                            ctx.config.ransac_num_iters,
                            ctx.config.ransac_max_inlier_error * ctx.zoom_ratio,
                            ctx.config.ransac_min_features,
                            noArray(),
                            HT_PNP_TYPE))
            return false;

		vector<Point2f> projected;

        vector<Point3f> all_points;
        
        for (int i = 0; i < ctx.config.max_keypoints; i++)
        {
            if (ctx.keypoints[i].idx == -1)
                continue;
            all_points.push_back(ctx.keypoint_uv[i]);
            
        }
        
		projectPoints(all_points, rvec, tvec, intrinsics, dist_coeffs, projected);

        mean_error = 0;
        float max_dist = ctx.config.ransac_max_reprojection_error * ctx.zoom_ratio;
        max_dist *= max_dist;
        int inliers_count = 0;
        
        std::vector<Point3f> final_3d;
        std::vector<Point2f> final_2d;
        
        for (int i = 0, j = 0; i < ctx.config.max_keypoints; i++)
        {
            if (ctx.keypoints[i].idx == -1)
                continue;
            
            float dist = ht_distance2d_squared(ctx.keypoints[i].position, projected[j++]);

            if (dist > max_dist)
            {
                ctx.keypoints[i].idx = -1;
                continue;
            }
            
            final_3d.push_back(ctx.keypoint_uv[i]);
            final_2d.push_back(ctx.keypoints[i].position);
            
            mean_error += dist;
            inliers_count++;
        }

        if (inliers_count >= 10)
        {
            mean_error = sqrt(mean_error / inliers_count);
            
            Mat rvec = Mat::zeros(3, 1, CV_64FC1);
            Mat tvec = Mat::zeros(3, 1, CV_64FC1);
            
            rvec.at<double> (0, 0) = 1.0;
            tvec.at<double> (0, 0) = 1.0;
            tvec.at<double> (1, 0) = 1.0;
            
            if (ctx.has_pose)
            {
                rvec = ctx.rvec.clone();
                tvec = ctx.tvec.clone();
            }
            
            if (!solvePnP(final_3d, final_2d, intrinsics, dist_coeffs, rvec, tvec, ctx.has_pose, HT_PNP_TYPE))
                return false;
            
            rvec_ = rvec.clone();
            tvec_ = tvec.clone();
            
            return true;
        }
    }

    if (ctx.config.debug)
    {
        fprintf(stderr, "ransac failed maxerr=%f zoom-ratio=%f cur=%d\n",
                ctx.config.ransac_max_inlier_error, ctx.zoom_ratio, (int) object_points.size());
        fflush(stderr);
    }

    return false;
}

