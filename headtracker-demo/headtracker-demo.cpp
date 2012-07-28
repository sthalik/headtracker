#include "stdafx.h"
#define HT_PI 3.14159265f
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	headtracker_t* ctx = ht_make_context(0);
	ht_result_t result;

	srand((int) getTickCount());

	cvNamedWindow("capture");

	IplImage* image = NULL;

	while (ht_cycle(ctx, &result)) {
		if (result.filled) {
			printf("%.3f %.2f | %.2f %.2f %.2f | %.1f %.1f %.1f\n",
				   result.confidence,
				   result.feature_ratio,
				   result.rotx * 180.0f / HT_PI,
				   result.roty * 180.0f / HT_PI,
				   result.rotz * 180.0f / HT_PI,
				   result.tx,
				   result.ty,
				   result.tz);
		}
		ht_frame_t frame = ht_get_bgr_frame(ctx);
		if (!image)
			image = cvCreateImage(cvSize(frame.width, frame.height), IPL_DEPTH_8U, 3);
		memcpy(image->imageData, frame.data, frame.width * frame.height * 3);
		cvShowImage("capture", image);
		cvWaitKey(1);
	}
	return 0;
}

