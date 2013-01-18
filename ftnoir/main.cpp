#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <tchar.h>
#include <ht-api.h>
#include "headtracker-ftnoir.h"

TCHAR shmName[] = TEXT(HT_SHM_NAME);
TCHAR mutexName[] = TEXT(HT_MUTEX_NAME);

int main(void)
{
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, shmName);

    if (!hMapFile)
    {
        printf("Can't create file mapping\n");
        return 1;
    }

    ht_shm_t* shm = (ht_shm_t*) MapViewOfFile(hMapFile,
                                              FILE_MAP_READ | FILE_MAP_WRITE,
                                              0,
                                              0,
                                              sizeof(ht_shm_t));

    if (!shm)
    {
        (void) CloseHandle(hMapFile);
        printf("Can't map view of file\n");
        return 1;
    }

    HANDLE hMutex = OpenMutex(SYNCHRONIZE, false, mutexName);

    if (!hMutex)
    {
        UnmapViewOfFile(shm);
        CloseHandle(hMapFile);
        printf("Can't open mutex\n");
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
        if (WaitForSingleObject(hMutex, INFINITE) == WAIT_OBJECT_0)
        {
            shm->result = result;
            ht_get_bgr_frame(ctx, &frame);
            if (frame.cols <= HT_MAX_VIDEO_WIDTH && frame.rows <= HT_MAX_VIDEO_HEIGHT && frame.channels <= HT_MAX_VIDEO_CHANNELS)
            {
                memcpy(shm->frame.frame, frame.data, shm->frame.width * shm->frame.height * shm->frame.channels);
                shm->frame.channels = frame.channels;
                shm->frame.width = frame.cols;
                shm->frame.height = frame.rows;
            }
            ReleaseMutex(hMutex);
            if (frame.data)
                delete[] frame.data;
        }
    }

    shm->running = false;

    ht_free_context(ctx);

    UnmapViewOfFile(shm);
    CloseHandle(hMapFile);
    CloseHandle(hMutex);

    return 0;
}
