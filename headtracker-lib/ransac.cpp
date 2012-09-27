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
                                        CvPOSITObject** prev_pObject,
                                        float* rotation_matrix,
                                        float* translation_vector) {
    double focal_length = ctx.focal_length;

    CvPOSITObject* posit_obj = cvCreatePOSITObject(model_points, point_cnt);
    if (*prev_pObject)
        cvReleasePOSITObject(prev_pObject);
    *prev_pObject = posit_obj;
    cvPOSIT(posit_obj,
            image_points,
            focal_length,
            cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, ctx.config.ransac_posit_iter, ctx.config.ransac_posit_eps),
            rotation_matrix,
            translation_vector);

    double foo = 0, bar = 0;
    for (int i = 0; i < point_cnt; i++) {
        double tmp = ht_distance2d_squared(ht_project_point(model_points[i], rotation_matrix, translation_vector, ctx.focal_length), image_points[i]);
        if (foo < tmp)
            foo = tmp;
        bar += tmp;
    }
    return sqrt(foo) * 0.8 + sqrt(bar / point_cnt) * 0.2;
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
               float max_error,
               int* best_keypoint_cnt,
               double* best_error,
               int* best_keypoints,
               float error_scale)
{
    if (ctx.keypoint_count == 0)
        return false;

    bool ret = false;
    int* keypoint_indices = new int[ctx.keypoint_count];
    int* model_keypoint_indices = new int[ctx.keypoint_count];
    const float bias = ctx.config.ransac_smaller_error_preference;
    const int K = ctx.config.ransac_num_iters;
    const int N = 4;

    *best_error = 1.0e20;
    *best_keypoint_cnt = 0;
    double best_score = -1;

    int kppos = 0;

    for (int i = 0; i < ctx.keypoint_count; i++) {
        if (ctx.keypoints[i].idx != -1)
            keypoint_indices[kppos++] = i;
    }

    CvPoint2D32f* image_points = new CvPoint2D32f[kppos];
    CvPoint3D32f* model_points = new CvPoint3D32f[kppos];
    CvPoint3D32f first_point = ctx.keypoint_uv[keypoint_indices[0]];

    if (kppos < N || kppos == 0) {
        goto end;
    }

    for (int iter = 0; iter < K; iter++) {
        ht_fisher_yates(keypoint_indices, kppos);
        int ipos = 0;

        float rotation_matrix[9];
        float translation_vector[3];

        memset(rotation_matrix, 0, sizeof(float) * 9);
        memset(translation_vector, 0, sizeof(float) * 3);

        double cur_error = ctx.config.max_best_error;

        CvPOSITObject* posit_obj = NULL;

        for (int kpos = 0; kpos < kppos; kpos++) {
            int idx = keypoint_indices[kpos];
            ht_keypoint& kp = ctx.keypoints[idx];
            model_points[ipos] = ctx.keypoint_uv[idx];
            model_points[ipos].x -= first_point.x;
            model_points[ipos].y -= first_point.y;
            model_points[ipos].z -= first_point.z;
            image_points[ipos] = kp.position;
            model_keypoint_indices[ipos] = idx;

            if (ipos - 1 >= N) {
                double e = ht_avg_reprojection_error(ctx, model_points, image_points, ipos+1, &posit_obj, rotation_matrix, translation_vector);
                e *= error_scale;
                if (e*max_error > cur_error)
                    continue;
                cur_error = e;
            }

            ipos++;

            double score = ipos * (1.0 - bias + bias * (ctx.config.max_best_error - cur_error) / ctx.config.max_best_error);

            if (ipos >= N &&
                score > best_score &&
                cur_error < ctx.config.max_best_error &&
                ipos >= ctx.config.ransac_min_features)
            {
                best_score = score;
                ret = true;
                *best_error = cur_error;
                *best_keypoint_cnt = ipos;
                for (int i = 0; i < ipos; i++)
                    best_keypoints[i] = model_keypoint_indices[i];
            }
        }

        if (posit_obj)
            cvReleasePOSITObject(&posit_obj);
    }

end:

    delete[] keypoint_indices;
    delete[] image_points;
    delete[] model_points;
    delete[] model_keypoint_indices;

    if (!ret && ctx.state == HT_STATE_TRACKING && ctx.abortp)
        abort();

    return ret;
}

class RansacThread : public QThread
{
public:
    void run();
    RansacThread(const headtracker_t& ctx,
                 float error_scale,
                 float max_error);
    int best_keypoint_cnt;
    double best_error;
    int* best_keypoints;
    bool ret;
    ~RansacThread() {
        delete[] best_keypoints;
    }
private:
    const headtracker_t& ctx;
    float error_scale;
    float max_error;
};

void RansacThread::run()
{
    ret = ht_ransac(ctx, max_error, &best_keypoint_cnt, &best_error, best_keypoints, error_scale);
}

RansacThread::RansacThread(const headtracker_t& ctx,
                           float error_scale,
                           float max_error)
    : ctx(ctx), error_scale(error_scale), max_error(max_error), ret(false)
{
    best_keypoints = new int[ctx.config.max_keypoints];
}

bool ht_ransac_best_indices(headtracker_t& ctx, double* best_error) {
    double max_error = ctx.config.ransac_max_error;
    bool ret = false;
    const int max_threads = ctx.config.ransac_max_threads;
    vector<RansacThread*> threads;
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = new RansacThread(ctx, ctx.zoom_ratio, max_error);
        t->start();
        threads.push_back(t);
    }
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = threads[i];
        t->wait();
    }
    int best = -1;
    double best_err = ctx.config.max_best_error;
    double best_score = -1;
    int best_cnt = 0;
    const double bias = ctx.config.ransac_smaller_error_preference;
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = threads[i];
        double score = t->best_keypoint_cnt * (1.0 - bias + bias * (ctx.config.max_best_error - t->best_error) / ctx.config.max_best_error);
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
