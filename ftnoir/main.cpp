#include <stdlib.h>
#include <stdio.h>
#include <ht-api.h>
#include "headtracker-ftnoir.h"
#include "compat.h"
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _MSC_VER
#    pragma comment(linker, "/subsystem:console /ENTRY:mainCRTStartup")
#endif

int main(void)
{
#ifdef _WIN32
    {
        int mask = 1 << 1;
        (void) SetProcessAffinityMask(GetCurrentProcess(), mask);
    }
#endif
	ht_shm_t* shm;
	PortableLockedShm lck_shm(HT_SHM_NAME, HT_MUTEX_NAME, sizeof(ht_shm_t));
	
	if ((shm = (ht_shm_t*) lck_shm.mem) == NULL || shm == (void*) -1)
	{
		fprintf(stderr, "Oh, bother\n");
		return 1;
	}

    shm->running = true;

    ht_result_t result;
    headtracker_t* ctx = ht_make_context(&shm->config, NULL);

    while (shm->timer++ < 200 && !shm->terminate)
    {
        if (shm->pause)
        {
            ht_reset(ctx);
            result.filled = false;
        }
        if (!ht_cycle(ctx, &result))
            break;
        lck_shm.lock();
		shm->result = result;
        const cv::Mat frame = ht_get_bgr_frame(ctx);
        if (frame.cols <= HT_MAX_VIDEO_WIDTH && frame.rows <= HT_MAX_VIDEO_HEIGHT && frame.channels() <= HT_MAX_VIDEO_CHANNELS)
		{
            const int cols = frame.cols;
            const int rows = frame.rows;
            const int pitch = cols * 3;
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    unsigned char* dest = &shm->frame.frame[y * pitch + 3 * x];
                    const cv::Vec3b& elt = frame.at<cv::Vec3b>(y, x);
                    const cv::Scalar elt2 = static_cast<cv::Scalar>(elt);
                    dest[0] = elt2.val[0];
                    dest[1] = elt2.val[1];
                    dest[2] = elt2.val[2];
                }
            }
            shm->frame.channels = frame.channels();
			shm->frame.width = frame.cols;
			shm->frame.height = frame.rows;
		}
		lck_shm.unlock();
    }

    shm->running = false;

    ht_free_context(ctx);

    return 0;
}
