#include <stdlib.h>
#include <stdio.h>
#include <ht-api.h>
#include "headtracker-ftnoir.h"
#include "compat.h"

int main(void)
{
	ht_shm_t* shm;
	PortableLockedShm lck_shm(HT_SHM_NAME, HT_MUTEX_NAME, sizeof(ht_shm_t));
	
	if ((shm = (ht_shm_t*) lck_shm.mem) == NULL)
	{
		fprintf(stderr, "Oh, bother\n");
		return 1;
	}

    shm->running = true;

    ht_result_t result;
    headtracker_t* ctx = ht_make_context(&shm->config, NULL);
    ht_frame_t frame;

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
		ht_get_bgr_frame(ctx, &frame);
		if (frame.cols <= HT_MAX_VIDEO_WIDTH && frame.rows <= HT_MAX_VIDEO_HEIGHT && frame.channels <= HT_MAX_VIDEO_CHANNELS)
		{
			memcpy(shm->frame.frame, frame.data, frame.cols * frame.rows * frame.channels);
			shm->frame.channels = frame.channels;
			shm->frame.width = frame.cols;
			shm->frame.height = frame.rows;
		}
		lck_shm.unlock();
		if (frame.data)
			delete[] frame.data;
    }

    shm->running = false;

    ht_free_context(ctx);

    return 0;
}
