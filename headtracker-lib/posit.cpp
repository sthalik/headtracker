#include "stdafx.h"

using namespace std;
using namespace cv;

// Copyright (C) 2000, Intel Corporation, all rights reserved.

struct CvPOSITObject
{
    int N;
    double* inv_matr;
    double* obj_vecs;
    double* img_vecs;
};

static  void  ht_icvReleasePOSITObject( CvPOSITObject ** ppObject )
{
    cvFree( ppObject );
}

static void
ht_icvPseudoInverse3D( double * __restrict a, double * __restrict b, int n, int method )
{
    int k;

    if( method == 0 )
    {
        double ata00 = 0;
        double ata11 = 0;
        double ata22 = 0;
        double ata01 = 0;
        double ata02 = 0;
        double ata12 = 0;
        double det = 0;

        /* compute matrix ata = transpose(a) * a  */
        for( k = 0; k < n; k++ )
        {
            double a0 = a[k];
            double a1 = a[n + k];
            double a2 = a[2 * n + k];

            ata00 += a0 * a0;
            ata11 += a1 * a1;
            ata22 += a2 * a2;

            ata01 += a0 * a1;
            ata02 += a0 * a2;
            ata12 += a1 * a2;
        }
        /* inverse matrix ata */
        {
            double inv_det;
            double p00 = ata11 * ata22 - ata12 * ata12;
            double p01 = -(ata01 * ata22 - ata12 * ata02);
            double p02 = ata12 * ata01 - ata11 * ata02;

            double p11 = ata00 * ata22 - ata02 * ata02;
            double p12 = -(ata00 * ata12 - ata01 * ata02);
            double p22 = ata00 * ata11 - ata01 * ata01;

            det += ata00 * p00;
            det += ata01 * p01;
            det += ata02 * p02;

            inv_det = 1 / det;

            /* compute resultant matrix */
            for( k = 0; k < n; k++ )
            {
                double a0 = a[k];
                double a1 = a[n + k];
                double a2 = a[2 * n + k];

                b[k] = (p00 * a0 + p01 * a1 + p02 * a2) * inv_det;
                b[n + k] = (p01 * a0 + p11 * a1 + p12 * a2) * inv_det;
                b[2 * n + k] = (p02 * a0 + p12 * a1 + p22 * a2) * inv_det;
            }
        }
    }

    /*if ( method == 1 )
       {
       }
     */

    return;
}

static  void  ht_icvPOSIT( CvPOSITObject *pObject, CvPoint2D32f * __restrict imagePoints,
                            double focalLength, CvTermCriteria criteria,
                            double* __restrict rotation, double* __restrict translation )
{
    int i, j, k;
    int count = 0, converged = 0;
    double inorm, jnorm, invInorm, invJnorm, invScale, scale = 0, inv_Z = 0;
    double diff = (double)criteria.epsilon;
    double inv_focalLength = 1 / focalLength;

    /* init variables */
    int N = pObject->N;
    double *objectVectors = pObject->obj_vecs;
    double *invMatrix = pObject->inv_matr;
    double *imgVectors = pObject->img_vecs;

    while( !converged )
    {
        if( count == 0 )
        {
            /* subtract out origin to get image vectors */
            for( i = 0; i < N; i++ )
            {
                imgVectors[i] = imagePoints[i + 1].x - imagePoints[0].x;
                imgVectors[N + i] = imagePoints[i + 1].y - imagePoints[0].y;
            }
        }
        else
        {
            diff = 0;
            /* Compute new SOP (scaled orthograthic projection) image from pose */
            for( i = 0; i < N; i++ )
            {
                /* objectVector * k */
                double old;
                double tmp = objectVectors[i] * rotation[6] /*[2][0]*/ +
                    objectVectors[N + i] * rotation[7]     /*[2][1]*/ +
                    objectVectors[2 * N + i] * rotation[8] /*[2][2]*/;

                tmp *= inv_Z;
                tmp += 1;

                old = imgVectors[i];
                imgVectors[i] = imagePoints[i + 1].x * tmp - imagePoints[0].x;

                diff = MAX( diff, (double) fabs( imgVectors[i] - old ));

                old = imgVectors[N + i];
                imgVectors[N + i] = imagePoints[i + 1].y * tmp - imagePoints[0].y;

                diff = MAX( diff, (double) fabs( imgVectors[N + i] - old ));
            }
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
        converged = ((criteria.type & CV_TERMCRIT_EPS) && (diff < criteria.epsilon));
        converged |= ((criteria.type & CV_TERMCRIT_ITER) && (count == criteria.max_iter));
    }
    invScale = 1 / scale;
    translation[0] = imagePoints[0].x * invScale;
    translation[1] = imagePoints[0].y * invScale;
    translation[2] = 1 / inv_Z;
}

static  void  ht_icvCreatePOSITObject( CvPoint3D32f * __restrict points,
                                    int numPoints,
                                    CvPOSITObject **ppObject )
{
    int i;

    /* Compute size of required memory */
    /* buffer for inverse matrix = N*3*float */
    /* buffer for storing weakImagePoints = numPoints * 2 * float */
    /* buffer for storing object vectors = N*3*float */
    /* buffer for storing image vectors = N*2*float */

    int N = numPoints - 1;
    int inv_matr_size = N * 3 * sizeof( double );
    int obj_vec_size = inv_matr_size;
    int img_vec_size = N * 2 * sizeof( double );
    CvPOSITObject *pObject;

    /* memory allocation */
    pObject = (CvPOSITObject *) cvAlloc( sizeof( CvPOSITObject ) +
                                         inv_matr_size + obj_vec_size + img_vec_size );

    /* part the memory between all structures */
    pObject->N = N;
    pObject->inv_matr = (double *) ((char *) pObject + sizeof( CvPOSITObject ));
    pObject->obj_vecs = (double *) ((char *) (pObject->inv_matr) + inv_matr_size);
    pObject->img_vecs = (double *) ((char *) (pObject->obj_vecs) + obj_vec_size);

/****************************************************************************************\
*          Construct object vectors from object points                                   *
\****************************************************************************************/
    for( i = 0; i < numPoints - 1; i++ )
    {
        pObject->obj_vecs[i] = points[i + 1].x - points[0].x;
        pObject->obj_vecs[N + i] = points[i + 1].y - points[0].y;
        pObject->obj_vecs[2 * N + i] = points[i + 1].z - points[0].z;
    }
/****************************************************************************************\
*   Compute pseudoinverse matrix                                                         *
\****************************************************************************************/
    ht_icvPseudoInverse3D( pObject->obj_vecs, pObject->inv_matr, N, 0 );

    *ppObject = pObject;
}

// end intel copyrighted code

bool ht_posit(CvPoint2D32f* image_points, CvPoint3D32f* model_points, int point_cnt, double* rotation_matrix, double* translation_vector, CvTermCriteria term_crit, double focal_length) {
	if (point_cnt < 4)
		return false;

    CvPOSITObject* posit_obj;

    ht_icvCreatePOSITObject(model_points, point_cnt, &posit_obj);
    ht_icvPOSIT(posit_obj, image_points, focal_length, term_crit, rotation_matrix, translation_vector);
	cvReleasePOSITObject(&posit_obj);

	return true;
}

ht_result_t ht_matrix_to_euler(double* rotation_matrix, double* translation_vector) {
	ht_result_t ret;

	if (rotation_matrix[0 * 3 + 2] > 0.9998) {
        ret.rotx = (HT_PI/2.0);
        ret.roty = atan2(rotation_matrix[1 * 3 + 0], rotation_matrix[1 * 3 + 1]);
		ret.rotz = 0.0f;
	} else if (rotation_matrix[0 * 3 + 2] < -0.9998) {
        ret.rotx = (HT_PI/-2.0);
        ret.roty = -atan2(rotation_matrix[1 * 3 + 0], rotation_matrix[1 * 3 + 1]);
		ret.rotz = 0.0f;
	} else {
        ret.rotx = asin(rotation_matrix[0 * 3 + 2]);
        ret.roty = -atan2(rotation_matrix[1 * 3 + 2], rotation_matrix[2 * 3 + 2]);
        ret.rotz = atan2(-rotation_matrix[0 * 3 + 1], rotation_matrix[0 * 3 + 0]);
	}

	ret.tx = translation_vector[0] / 10.0f;
	ret.ty = translation_vector[1] / 10.0f;
	ret.tz = translation_vector[2] / 10.0f;

	return ret;
}
