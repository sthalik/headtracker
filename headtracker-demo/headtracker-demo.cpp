#include "stdafx.h"
#define HT_PI 3.14159265f
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static volatile bool ht_quitp = false;

#ifdef __unix
#include <signal.h>

static void ht_quit_handler(int foo) {
    ht_quitp = true;
}

#endif

int main(int argc, char** argv)
{
    srand(0);
    cv::setNumThreads(4);
    bool start = false;
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

    headtracker_t* ctx = ht_make_context(&conf, argc > 1 ? argv[1] : NULL);
    ht_result_t result;

    cvNamedWindow("capture");

    while (!ht_quitp && ht_cycle(ctx, &result)) {
        if (result.filled) {
            start = true;
#if 1
			printf("%.3f | %.2f %.2f %.2f | %.1f %.1f %.1f\n",
				   result.confidence,
				   result.rotx * 180.0f / HT_PI,
				   result.roty * 180.0f / HT_PI,
				   result.rotz * 180.0f / HT_PI,
				   result.tx,
				   result.ty,
                   result.tz);
#endif
        } else if (start && argc > 1) {
            abort();
            break;
        }

        ht_frame_t frame = ht_get_bgr_frame(ctx);
        imshow("capture", frame.data);
        waitKey(1);
    }
        ht_free_context(ctx);
	return 0;
}

