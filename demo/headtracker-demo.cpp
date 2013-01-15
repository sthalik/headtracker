#define HT_PI 3.14159265f
#include "headtracker.h"
#include <opencv2/opencv.hpp>
#include <QtNetwork/QUdpSocket>

using namespace std;
using namespace cv;

static volatile bool ht_quitp = false;

#ifdef __unix
#include <signal.h>

static void ht_quit_handler(int foo) {
    ht_quitp = true;
}

#endif

#pragma pack(push, 2)
struct THeadPoseData {
        double x, y, z, yaw, pitch, roll;
        long frame_number;
};
#pragma pack(pop)

int main(int argc, char** argv)
{
    bool start = false;
#ifdef __unix
    (void) signal(SIGTERM, ht_quit_handler);
    (void) signal(SIGHUP, ht_quit_handler);
    (void) signal(SIGINT, ht_quit_handler);
#endif
	ht_config_t conf;
	FILE* cfg;

    ht_make_config(&conf);

	if ((cfg = fopen("config.txt", "r")) != NULL) {
        ht_load_config(cfg, &conf);
		fclose(cfg);
    }

    headtracker_t* ctx = ht_make_context(&conf, argc > 1 ? argv[1] : NULL);
    ht_result_t result;

    QUdpSocket sock;
    QHostAddress addr("127.0.0.1");
    sock.bind(addr, QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
    THeadPoseData pose;
    int frameno = 0;

    namedWindow("capture");

    while (!ht_quitp && ht_cycle(ctx, &result)) {
        if (result.filled) {
            start = true;
            pose.frame_number = frameno;
            pose.x = result.tx;
            pose.y = result.ty;
            pose.z = result.tz;
            pose.yaw = result.rotx * 180.0f / HT_PI;
            pose.pitch = result.roty * 180.0f / HT_PI;
            pose.roll = result.rotz * 180.0f / HT_PI;
            sock.writeDatagram((const char*) &pose, sizeof(THeadPoseData), addr, 5550);
#if 0
            printf("POSE %.2f %.2f %.2f | %.2f %.2f %.2f\n",
                   pose.yaw, pose.pitch, pose.roll,
                   pose.x, pose.y, pose.z);
#endif
        } else if (start && argc > 1) {
            abort();
            break;
        }
        frameno++;
        ht_frame_t frame;
        ht_get_bgr_frame(ctx, &frame);
        if (frame.data) {
            Mat foo(frame.rows, frame.cols, CV_8UC3, frame.data);
            imshow("capture", foo);
            delete[] frame.data;
        }
        waitKey(1);
    }

    ht_free_context(ctx);
	return 0;
}

