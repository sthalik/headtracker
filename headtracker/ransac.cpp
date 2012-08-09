#include "stdafx.h"

using namespace std;
using namespace cv;

struct CvPOSITObject
{
    int N;
    float* inv_matr;
    float* obj_vecs;
    float* img_vecs;
};

static void ht_icvRecreatePOSITObject(const CvPOSITObject *prev_pObject,
                                      CvPOSITObject *new_ppObject,
									  const CvPoint2D32f *imagePoints)
{
	int prev_N;
	int N = new_ppObject->N;

	if (prev_pObject == NULL) {
		prev_N = 0;
	} else {
		prev_N = prev_pObject->N;
		for (int i = 0; i < prev_N; i++) {
			new_ppObject->img_vecs[i] = prev_pObject->img_vecs[i];
			new_ppObject->img_vecs[N + i] = prev_pObject->img_vecs[N + i];
		}
	}

	for (int i = prev_N; i < N; i++) {
		new_ppObject->img_vecs[i] = imagePoints[i + 1].x - imagePoints[0].x;
		new_ppObject->img_vecs[N + i] = imagePoints[i + 1].y - imagePoints[0].y;
	}
}

// original POSIT code, modified, copyright/authorship applies

static void ht_icvPOSIT( CvPOSITObject *pObject, CvPoint2D32f *imagePoints,
                         float focalLength, float max_eps, int max_iter,
                         float* rotation, float* translation )
{
    int i, j, k;
    int count = 0, converged = 0;
    float inorm, jnorm, invInorm, invJnorm, invScale, scale = 0, inv_Z = 0;
    float diff = max_eps;
    float inv_focalLength = 1 / focalLength;

    /* init variables */
    int N = pObject->N;
    float *objectVectors = pObject->obj_vecs;
    float *invMatrix = pObject->inv_matr;
    float *imgVectors = pObject->img_vecs;

    while( !converged )
    {
        diff = 0;
        /* Compute new SOP (scaled orthograthic projection) image from pose */
        for( i = 0; i < N; i++ )
        {
            /* objectVector * k */
            float old;
            float tmp = objectVectors[i] * rotation[6] /*[2][0]*/ +
                objectVectors[N + i] * rotation[7]     /*[2][1]*/ +
                objectVectors[2 * N + i] * rotation[8] /*[2][2]*/;

            tmp *= inv_Z;
            tmp += 1;

            old = imgVectors[i];
            imgVectors[i] = imagePoints[i + 1].x * tmp - imagePoints[0].x;

            diff = MAX( diff, (float) fabs( imgVectors[i] - old ));

            old = imgVectors[N + i];
            imgVectors[N + i] = imagePoints[i + 1].y * tmp - imagePoints[0].y;

            diff = MAX( diff, (float) fabs( imgVectors[N + i] - old ));
        }
        /* calculate I and J vectors */
        for( i = 0; i < 2; i++ )
        {
            for( j = 0; j < 3; j++ )
            {
                rotation[3*i+j] /*[i][j]*/ = 0;
                for( k = 0; k < N; k++ )
                {
                    rotation[3*i+j] /*[i][j]*/ += invMatrix[j * N + k] * imgVectors[i * N + k];
                }
            }
        }

        inorm = rotation[0] /*[0][0]*/ * rotation[0] /*[0][0]*/ +
                rotation[1] /*[0][1]*/ * rotation[1] /*[0][1]*/ + 
                rotation[2] /*[0][2]*/ * rotation[2] /*[0][2]*/;

        jnorm = rotation[3] /*[1][0]*/ * rotation[3] /*[1][0]*/ +
                rotation[4] /*[1][1]*/ * rotation[4] /*[1][1]*/ + 
                rotation[5] /*[1][2]*/ * rotation[5] /*[1][2]*/;

        invInorm = cvInvSqrt( inorm );
        invJnorm = cvInvSqrt( jnorm );

        inorm *= invInorm;
        jnorm *= invJnorm;

        rotation[0] /*[0][0]*/ *= invInorm;
        rotation[1] /*[0][1]*/ *= invInorm;
        rotation[2] /*[0][2]*/ *= invInorm;

        rotation[3] /*[1][0]*/ *= invJnorm;
        rotation[4] /*[1][1]*/ *= invJnorm;
        rotation[5] /*[1][2]*/ *= invJnorm;

        /* row2 = row0 x row1 (cross product) */
        rotation[6] /*->m[2][0]*/ = rotation[1] /*->m[0][1]*/ * rotation[5] /*->m[1][2]*/ -
                                    rotation[2] /*->m[0][2]*/ * rotation[4] /*->m[1][1]*/;
       
        rotation[7] /*->m[2][1]*/ = rotation[2] /*->m[0][2]*/ * rotation[3] /*->m[1][0]*/ -
                                    rotation[0] /*->m[0][0]*/ * rotation[5] /*->m[1][2]*/;
       
        rotation[8] /*->m[2][2]*/ = rotation[0] /*->m[0][0]*/ * rotation[4] /*->m[1][1]*/ -
                                    rotation[1] /*->m[0][1]*/ * rotation[3] /*->m[1][0]*/;

        scale = (inorm + jnorm) / 2.0f;
        inv_Z = scale * inv_focalLength;

        count++;
        converged = diff < max_eps || count == max_iter;
    }
    invScale = 1 / scale;
    translation[0] = imagePoints[0].x * invScale;
    translation[1] = imagePoints[0].y * invScale;
    translation[2] = 1 / inv_Z;
}

static error_t ht_avg_reprojection_error(headtracker_t& ctx,
										 CvPoint3D32f* model_points,
										 CvPoint2D32f* image_points,
										 int point_cnt,
										 CvPOSITObject** prev_pObject,
										 float* rotation_matrix,
										 float* translation_vector) {
	double focal_length = ctx.focal_length;

	error_t ret;

	CvPOSITObject* posit_obj = cvCreatePOSITObject(model_points, point_cnt);
	ht_icvRecreatePOSITObject(*prev_pObject, posit_obj, image_points);
	if (*prev_pObject)
		cvReleasePOSITObject(prev_pObject);
	*prev_pObject = posit_obj;
	ht_icvPOSIT(posit_obj,
				image_points,
				focal_length,
				ctx.config.ransac_posit_eps,
				5 + point_cnt * 0.666,
				rotation_matrix,
				translation_vector);

	float avg = 0;

	for (int i = 0; i < point_cnt; i++)
		avg += ht_distance2d_squared(ht_project_point(model_points[i], rotation_matrix, translation_vector, ctx.focal_length), image_points[i]);

	ret.avg = (float) sqrt(avg / point_cnt);

	return ret;
}

void ht_fisher_yates(int* indices, int count) {
	int tmp;

	for (int i = count - 1; i > 0; i--) {
		int j = rand() % i;
		tmp = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp;
	}
}

bool ht_ransac(headtracker_t& ctx,
			   int max_iter,
			   float max_error,
			   int min_consensus,
			   int* best_feature_cnt,
			   int* best_keypoint_cnt,
			   error_t* best_error,
			   int* best_indices,
			   int* best_keypoints,
			   float error_scale)
{
	if (ctx.keypoint_count == 0)
		return false;

	int mcnt = ctx.model.count;
	int k = 0;
	bool ret = false;
	float max_consensus_error = ctx.config.ransac_max_consensus_error * error_scale;
	int* keypoint_indices = new int[ctx.keypoint_count];
	int* model_feature_indices = new int[mcnt];
	int* model_keypoint_indices = new int[ctx.keypoint_count];
	int* indices = new int[mcnt];
	const float bias = ctx.config.ransac_smaller_error_preference;

	best_error->avg = 1.0e10;
	*best_feature_cnt = 0;
	*best_keypoint_cnt = 0;

	for (int i = 0; i < mcnt; i++) {
		if (ctx.features[i].x != -1)
			indices[k++] = i;
	}

	int kppos = 0;

	for (int i = 0; i < ctx.keypoint_count; i++) {
		if (ctx.keypoints[i].idx != -1)
			keypoint_indices[kppos++] = i;
	}

	CvPoint2D32f* image_points = new CvPoint2D32f[k + kppos];
	CvPoint3D32f* model_points = new CvPoint3D32f[k + kppos];

	if ((k + kppos) < min_consensus || (k + kppos) < 4)
		goto end;

	for (int iter = 0; iter < max_iter; iter++) {
		ht_fisher_yates(indices, k);
		ht_fisher_yates(keypoint_indices, kppos);
		int ipos = 0;
		int fpos = 0;
		int kpos = 0;
		int gfpos = 0;
		int gkpos = 0;
		float rotation_matrix[9];
		float translation_vector[3];
		float rotation_new[9];
		float translation_new[3];

		memset(rotation_matrix, 0, sizeof(float) * 9);
		memset(translation_vector, 0, sizeof(float) * 3);

		CvPoint3D32f first_point = ctx.feature_uv[indices[0]];

		error_t cur_error;
		cur_error.avg = max_consensus_error;

		CvPOSITObject* posit_obj = NULL;

		for (; fpos < k; fpos++) {
			int idx = indices[fpos];
			model_points[ipos] = ctx.feature_uv[idx];
			model_points[ipos].x -= first_point.x;
			model_points[ipos].y -= first_point.y;
			model_points[ipos].z -= first_point.z;
			image_points[ipos] = ctx.features[idx];
			model_feature_indices[gfpos] = idx;

			if (ipos >= ctx.config.ransac_min_features) {
				memcpy(rotation_new, rotation_matrix, sizeof(float) * 9);
				memcpy(translation_new, translation_vector, sizeof(float) * 3);
				error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, ipos+1, &posit_obj, rotation_new, translation_new);
				e.avg *= error_scale;

				if (e.avg*max_error > cur_error.avg)
					continue;
				cur_error.avg = e.avg;

				memcpy(rotation_matrix, rotation_new, sizeof(float) * 9);
				memcpy(translation_vector, translation_new, sizeof(float) * 3);
			}

			ipos++;
			gfpos++;

			if (ipos >= min_consensus && ipos > *best_feature_cnt + *best_keypoint_cnt) {
				ret = true;
				*best_error = cur_error;
				*best_feature_cnt = gfpos;
				*best_keypoint_cnt = gkpos;
				for (int i = 0; i < gfpos; i++)
					best_indices[i] = model_feature_indices[i];
				for (int i = 0; i < gkpos; i++)
					best_keypoints[i] = model_keypoint_indices[i];
			}
		}

		for (; kpos < kppos; kpos++) {
			int idx = keypoint_indices[kpos];
			ht_keypoint& kp = ctx.keypoints[idx];
			model_points[ipos] = ctx.keypoint_uv[idx];
			model_points[ipos].x -= first_point.x;
			model_points[ipos].y -= first_point.y;
			model_points[ipos].z -= first_point.z;
			image_points[ipos] = kp.position;
			model_keypoint_indices[gkpos] = idx;
			if (ipos >= ctx.config.ransac_min_features) {
				memcpy(rotation_new, rotation_matrix, sizeof(float) * 9);
				memcpy(translation_new, translation_vector, sizeof(float) * 3);
				error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, ipos+1, &posit_obj, rotation_new, translation_new);
				e.avg *= error_scale;
				if (e.avg*max_error > cur_error.avg)
					continue;
				cur_error.avg = e.avg;
				memcpy(rotation_matrix, rotation_new, sizeof(float) * 9);
				memcpy(translation_vector, translation_new, sizeof(float) * 3);
			}

			ipos++;
			gkpos++;

			if (ipos >= min_consensus &&
				ipos * ((1.0f - bias) + bias * (best_error->avg / cur_error.avg)) > *best_feature_cnt + *best_keypoint_cnt)
			{
				ret = true;
				*best_error = cur_error;
				*best_feature_cnt = gfpos;
				*best_keypoint_cnt = gkpos;
				for (int i = 0; i < gfpos; i++)
					best_indices[i] = model_feature_indices[i];
				for (int i = 0; i < gkpos; i++)
					best_keypoints[i] = model_keypoint_indices[i];
			}
		}
		if (posit_obj)
			cvReleasePOSITObject(&posit_obj);
	}

end:

	delete[] keypoint_indices;
	delete[] indices;
	delete[] image_points;
	delete[] model_points;
	delete[] model_feature_indices;
	delete[] model_keypoint_indices;

	return ret;
}

bool ht_ransac_best_indices(headtracker_t& ctx, error_t* best_error) {
	int min_features = ctx.state == HT_STATE_TRACKING ? ctx.feature_count * ctx.config.ransac_min_consensus : ctx.config.min_track_start_features;
	int* best_feature_indices = new int[ctx.feature_count];
	int* best_keypoint_indices = new int[ctx.keypoint_count];
	int best_feature_cnt, best_keypoint_cnt;
	if (ht_ransac(ctx,
				  ctx.config.ransac_iter,
				  ctx.config.ransac_max_error,
				  max(min_features, ctx.config.ransac_min_features),
				  &best_feature_cnt,
				  &best_keypoint_cnt,
				  best_error,
				  best_feature_indices,
				  best_keypoint_indices,
				  ctx.zoom_ratio)) {
		char* fusedp = new char[ctx.model.count];
		char* kusedp = new char[ctx.config.max_keypoints];
		for (int i = 0; i < ctx.model.count; i++)
			fusedp[i] = 0;
		for (int i = 0; i < best_feature_cnt; i++) {
			fusedp[best_feature_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.model.count; i++) {
			if (!fusedp[i]) {
				if (ctx.features[i].x != -1) {
					if (++ctx.feature_failed_iters[i] > ctx.config.feature_max_failed_ransac) {
						ctx.features[i].x = -1;
						ctx.feature_count--;
					}
				}
			} else {
				ctx.feature_failed_iters[i] = 0;
			}
		}
		for (int i = 0; i < ctx.config.max_keypoints; i++)
			kusedp[i] = 0;
		for (int i = 0; i < best_keypoint_cnt; i++)
			kusedp[best_keypoint_indices[i]] = 1;
		for (int i = 0; i < ctx.config.max_keypoints; i++) {
			if (!kusedp[i] && ctx.keypoints[i].idx != -1) {
				if (++ctx.keypoint_failed_iters[i] > ctx.config.keypoint_max_failed_ransac) {
					ctx.keypoints[i].idx = -1;
					ctx.keypoint_count--;
				}
			} else {
				ctx.keypoint_failed_iters[i] = 0;
			}
		}
		delete[] fusedp;
		delete[] kusedp;
		delete[] best_keypoint_indices;
		delete[] best_feature_indices;
		return true;
	}
	delete[] best_keypoint_indices;
	delete[] best_feature_indices;
	return false;
}