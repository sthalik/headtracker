#include "stdafx.h"
#define HT_PI 3.14159265f
#include <opencv2/opencv.hpp>

#define FRAMESKIP 0

using namespace std;
using namespace cv;

static volatile bool ht_quitp = false;

#ifdef __unix
#include <signal.h>

static void ht_quit_handler(int foo) {
    ht_quitp = true;
}

#endif

int main(int, char**)
{
#ifdef __unix
    (void) signal(SIGTERM, ht_quit_handler);
    (void) signal(SIGHUP, ht_quit_handler);
    (void) signal(SIGINT, ht_quit_handler);
#endif
	ht_config_t conf;
	FILE* cfg;

	if ((cfg = fopen("config.txt", "r")) != NULL) {
		conf = ht_load_config(cfg);
		fclose(cfg);
	} else {
		conf = ht_make_config();
	}

	headtracker_t* ctx = ht_make_context(&conf);
	ht_result_t result;

	srand((int) getTickCount());

	cvNamedWindow("capture");

	IplImage* image = NULL;
    unsigned char framecnt = 0;

    while (!ht_quitp && ht_cycle(ctx, &result)) {
		if (result.filled) {
			printf("%.3f | %.2f %.2f %.2f | %.1f %.1f %.1f\n",
				   result.confidence,
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
        if (framecnt++ == FRAMESKIP) {
            framecnt = 0;
            cvShowImage("capture", image);
        }
        cvWaitKey(1);
	}
	return 0;
}

