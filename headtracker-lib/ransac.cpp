#include "stdafx.h"
#include <limits.h>

#include <Qt>
#include <QtGlobal>
#include <QThread>

using namespace std;
using namespace cv;

static double ht_avg_reprojection_error(const headtracker_t& ctx,
                                        CvPoint3D32f* model_points,
                                        CvPoint2D32f* image_points,
                                        int point_cnt,
                                        double* rotation_matrix,
                                        double* translation_vector) {
    ht_posit(image_points,
             model_points,
             point_cnt,
             rotation_matrix,
             translation_vector,
             cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,
                            ctx.config.ransac_posit_iter,
                            ctx.config.ransac_posit_eps),
             ctx.focal_length);

    double bar = 0;
    for (int i = 0; i < point_cnt; i++) {
        bar += ht_distance2d_squared(ht_project_point(model_points[i], rotation_matrix, translation_vector, ctx.focal_length),
                                     image_points[i]);
    }
    return sqrt(bar / point_cnt);
}

void ht_fisher_yates(int* indices, int count) {
    int tmp;

    for (int i = count - 1; i > 0; i--) {
        int j = qrand() % i;
        tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

bool ht_ransac(const headtracker_t& ctx,
               int* best_keypoint_cnt,
               double* best_error,
               int* best_keypoints)
{
    if (ctx.keypoint_count == 0)
        return false;

    bool ret = false;
    int* keypoint_indices = new int[ctx.keypoint_count];
    int* orig_indices = new int[ctx.keypoint_count];
    int* model_keypoint_indices = new int[ctx.keypoint_count];
    const double bias = ctx.config.ransac_smaller_error_preference;
    const int K = ctx.config.ransac_num_iters;
    const int N = 5;

    *best_error = 1.0e20;
    *best_keypoint_cnt = 0;
    double best_score = -1;

    int kppos = 0;

    for (int i = 0; i < ctx.keypoint_count; i++) {
        if (ctx.keypoints[i].idx != -1)
            orig_indices[kppos++] = i;
    }

    CvPoint2D32f* image_points = new CvPoint2D32f[kppos];
    CvPoint3D32f* model_points = new CvPoint3D32f[kppos];

    double rotation_matrix[9];
    double translation_vector[3];
    double max_max_error = (ctx.config.ransac_max_error - ctx.config.ransac_avg_error) / (ctx.config.max_keypoints - N);

    if (kppos < N || kppos == 0) {
        goto end;
    }

    for (int iter = 0; iter < K; iter++) {
        memcpy(keypoint_indices, orig_indices, sizeof(int) * kppos);
        ht_fisher_yates(keypoint_indices, kppos);
        CvPoint3D32f first_point = ctx.keypoint_uv[keypoint_indices[0]];
        int ipos = 0;
        double avg_error = 1.0e20;

        for (int kpos = 0; kpos < kppos; kpos++) {
            double max_avg_error = ctx.config.ransac_avg_error;
            int idx = keypoint_indices[kpos];
            ht_keypoint& kp = ctx.keypoints[idx];
            model_points[ipos] = ctx.keypoint_uv[idx];
            model_points[ipos].x -= first_point.x;
            model_points[ipos].y -= first_point.y;
            model_points[ipos].z -= first_point.z;
            image_points[ipos] = kp.position;
            model_keypoint_indices[ipos] = idx;

            if (ipos - 1 >= N) {
                double e = ht_avg_reprojection_error(ctx,
                                                     model_points,
                                                     image_points,
                                                     ipos+1,
                                                     rotation_matrix,
                                                     translation_vector);
                if (e > ctx.config.max_best_error && ipos > ctx.config.ransac_min_features)
                    goto end2;
                if (e*max_avg_error > avg_error)
                    continue;
                avg_error = e;
                max_avg_error += max_max_error;
            }

            ipos++;

            double score = ipos * (1.0 - bias) / ctx.config.max_keypoints + bias * (ctx.config.max_best_error - avg_error) / ctx.config.max_best_error;

            if (ipos >= N &&
                score > best_score &&
                avg_error < ctx.config.max_best_error &&
                ipos >= ctx.config.ransac_min_features)
            {
                best_score = score;
                ret = true;
                *best_error = avg_error;
                *best_keypoint_cnt = ipos;
                for (int i = 0; i < ipos; i++)
                    best_keypoints[i] = model_keypoint_indices[i];
            }
        }
end2:
        ;
    }

end:

    if (!ret && ctx.state == HT_STATE_TRACKING && ctx.abortp)
        abort();

    delete[] keypoint_indices;
    delete[] image_points;
    delete[] model_points;
    delete[] model_keypoint_indices;
    delete[] orig_indices;

    return ret;
}

class RansacThread : public QThread
{
public:
    void run();
    RansacThread(const headtracker_t& ctx);
    int best_keypoint_cnt;
    double best_error;
    int* best_keypoints;
    bool ret;
    ~RansacThread() {
        delete[] best_keypoints;
    }
private:
    const headtracker_t& ctx;
};

void RansacThread::run()
{
    ret = ht_ransac(ctx, &best_keypoint_cnt, &best_error, best_keypoints);
}

RansacThread::RansacThread(const headtracker_t& ctx) :
    ctx(ctx),
    ret(false)
{
    best_keypoints = new int[ctx.config.max_keypoints];
}

bool ht_ransac_best_indices(headtracker_t& ctx, double* best_error) {
    bool ret = false;
    const int max_threads = ctx.config.ransac_max_threads;
    vector<RansacThread*> threads;
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = new RansacThread(ctx);
        t->start();
        threads.push_back(t);
    }
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = threads[i];
        t->wait();
    }
    int best = -1;
    double best_err = ctx.config.max_best_error;
    double best_score = -1e9;
    int best_cnt = 0;
    const double bias = ctx.config.ransac_smaller_error_preference;
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = threads[i];
        double score = t->best_keypoint_cnt * (1.0 - bias) / ctx.config.max_keypoints + bias * (ctx.config.max_best_error - t->best_error) / ctx.config.max_best_error;
        if (t->ret && score > best_score) {
            best = i;
            best_err = t->best_error;
            best_cnt = t->best_keypoint_cnt;
        }
    }
    if (best != -1) {
        int best_keypoint_cnt = threads[best]->best_keypoint_cnt;
        const int* best_keypoint_indices = threads[best]->best_keypoints;
        char* kusedp = new char[ctx.config.max_keypoints];
        for (int i = 0; i < ctx.config.max_keypoints; i++)
            kusedp[i] = 0;
        for (int i = 0; i < best_keypoint_cnt; i++)
            kusedp[best_keypoint_indices[i]] = 1;
        for (int i = 0; i < ctx.config.max_keypoints; i++) {
            if (!kusedp[i] && ctx.keypoints[i].idx != -1) {
                ctx.keypoints[i].idx = -1;
                ctx.keypoint_count--;
            }
        }
        delete[] kusedp;
        ret = true;
        *best_error = best_err;
    }
    while(!threads.empty()) {
        delete threads.back();
        threads.pop_back();
    }

    return ret;
}
