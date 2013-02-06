/* Copyright (c) 2013 Stanisław Halik <sthalik@misaki.pl>

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */
#pragma once

#if defined(_WIN32) || defined(__WIN32)
#include <windows.h>
#else
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#endif

class PortableLockedShm {
public:
    PortableLockedShm(const char *shmName, const char *mutexName, int mapSize);
    ~PortableLockedShm();
    void lock();
    void unlock();
    void* mem;
private:
#if defined(_WIN32) || defined(__WIN32)
    HANDLE hMutex, hMapFile;
#else
    int fd, size;
    //char shm_filename[NAME_MAX];
#endif
};
