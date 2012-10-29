#include "stdafx.h"
#include <limits.h>

#include <Qt>
#include <QtGlobal>
#include <QThread>

using namespace std;
using namespace cv;

static float ht_avg_reprojection_error(const headtracker_t& ctx,
                                        CvPoint3D32f* model_points,
                                        CvPoint2D32f* image_points,
                                        int point_cnt,
                                        float* rotation_matrix,
                                        float* translation_vector)
{
    ht_posit(image_points,
             model_points,
             point_cnt,
             rotation_matrix,
             translation_vector,
             cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,
                            ctx.config.ransac_posit_iter,
                            ctx.config.ransac_posit_eps),
             ctx.focal_length);

    float bar = 0;
    for (int i = 0; i < point_cnt; i++) {
        bar += ht_distance2d_squared(ht_project_point(model_points[i],
                                                      rotation_matrix,
                                                      translation_vector,
                                                      ctx.focal_length),
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
               float* best_error,
               int* best_keypoints)
{
    int total = ctx.keypoint_count;

    if (total < 4)
        return false;

    bool ret = false;
    int* keypoint_indices = new int[total];
    int* orig_indices = new int[total];
    const int K = ctx.config.ransac_num_iters;
    const int N = 5;

    *best_error = 1e20;

    int kppos = 0;

    for (int i = 0; i < total; i++) {
        if (ctx.keypoints[i].idx != -1)
            orig_indices[kppos++] = i;
    }

    CvPoint2D32f* image_points = new CvPoint2D32f[kppos+1];
    CvPoint3D32f* model_points = new CvPoint3D32f[kppos+1];

    float rotation_matrix[9];
    float translation_vector[3];
    float max_avg_error = ctx.config.ransac_avg_error;
    float minf = ctx.config.ransac_min_features * kppos;

    CvPoint3D32f* best_obj_points = new CvPoint3D32f[kppos];
    CvPoint2D32f* best_img_points = new CvPoint2D32f[kppos];
    int best_count;
    CvPoint3D32f pivot;

    if (kppos >= N && kppos >= minf) {
        for (int iter = 0; iter < K; iter++) {
            memcpy(keypoint_indices, orig_indices, sizeof(int) * kppos);
            ht_fisher_yates(keypoint_indices, kppos);
            CvPoint3D32f first_point = ctx.keypoint_uv[keypoint_indices[0]];
            int ipos = 0;
            float avg_error = 1.0e20;

            for (int kpos = 0; kpos < kppos; kpos++) {
                int idx = keypoint_indices[kpos];
                ht_keypoint& kp = ctx.keypoints[idx];
                model_points[ipos] = ctx.keypoint_uv[idx];
                model_points[ipos].x -= first_point.x;
                model_points[ipos].y -= first_point.y;
                model_points[ipos].z -= first_point.z;
                image_points[ipos] = kp.position;

                if (ipos - 1 >= N && (ipos & 1) == 0) {
                    float e = ht_avg_reprojection_error(ctx,
                                                         model_points,
                                                         image_points,
                                                         ipos+1,
                                                         rotation_matrix,
                                                         translation_vector);
                    if (e*max_avg_error > avg_error) {
                        ipos--;
                        continue;
                    }
                    avg_error = e;
                }

                ipos++;

                if (avg_error < *best_error &&
                    ipos >= minf)
                {
                    *best_error = avg_error;
                    ret = true;
                    memcpy(best_img_points, image_points, ipos * sizeof(CvPoint2D32f));
                    memcpy(best_obj_points, model_points, ipos * sizeof(CvPoint3D32f));
                    best_count = ipos;
                    pivot = first_point;
                }

                if (ipos >= minf)
                    break;
            }
        }
    }

    if (!ret && ctx.state == HT_STATE_TRACKING && ctx.abortp)
        abort();

    if (ret) {
        float f = ctx.focal_length;
        ret = ht_posit(best_img_points,
                       best_obj_points,
                       best_count,
                       rotation_matrix,
                       translation_vector,
                       cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_NUMBER, 200, 1e-8),
                       f);
        if (ret) {
            float max_error = ctx.config.ransac_max_error * ctx.zoom_ratio;
            int j = 0;
            float cur_error = 0;
            max_error *= max_error;
            for (int i = 0; i < kppos; i++) {
                int idx = orig_indices[i];
                CvPoint3D32f pt = ctx.keypoint_uv[idx];
                CvPoint2D32f pt2d = ctx.keypoints[idx].position;
                CvPoint2D32f projection = ht_project_point(cvPoint3D32f(pt.x - pivot.x,
                                                                        pt.y - pivot.y,
                                                                        pt.z - pivot.z),
                                                           rotation_matrix,
                                                           translation_vector,
                                                           f);
                float error = ht_distance2d_squared(projection, pt2d);
                if (error > max_error)
                    continue;
                best_keypoints[j++] = idx;
                cur_error += error;
            }
            *best_keypoint_cnt = j;
            *best_error = sqrt(cur_error / j);
            ret = *best_error < ctx.config.max_best_error * ctx.zoom_ratio;
        }
    }

    delete[] keypoint_indices;
    delete[] image_points;
    delete[] model_points;
    delete[] orig_indices;
    delete[] best_img_points;
    delete[] best_obj_points;

    return ret;
}

class RansacThread : public QThread
{
public:
    void run();
    RansacThread(const headtracker_t& ctx);
    int best_keypoint_cnt;
    float best_error;
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

bool ht_ransac_best_indices(headtracker_t& ctx, float* best_error) {
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
    float best_err = 1.0e20;
    for (int i = 0; i < max_threads; i++) {
        RansacThread* t = threads[i];
        if (t->ret && t->best_error < best_err) {
            best = i;
            best_err = t->best_error;
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
