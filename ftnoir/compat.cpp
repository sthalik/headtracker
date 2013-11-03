/* Copyright (c) 2013 Stanisław Halik <sthalik@misaki.pl>

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */

#include "compat.h"
#include <stdio.h>
#include <limits.h>
#include <string>
#include <sstream>
#include <errno.h>

#ifndef NAME_MAX
#	include <stdio.h>
#	define NAME_MAX FILENAME_MAX
#endif

#if defined(_WIN32) || defined(__WIN32)

PortableLockedShm::PortableLockedShm(const char* shmName, const char* mutexName, int mapSize)
{
    hMutex = CreateMutexA(NULL, false, mutexName);
    hMapFile = CreateFileMappingA(
                 INVALID_HANDLE_VALUE,
                 NULL,
                 PAGE_READWRITE,
                 0,
                 mapSize,
                 shmName);
    mem = MapViewOfFile(hMapFile,
                        FILE_MAP_WRITE,
                        0,
                        0,
                        mapSize);
}

PortableLockedShm::~PortableLockedShm()
{
    UnmapViewOfFile(mem);
    CloseHandle(hMapFile);
    CloseHandle(hMutex);
}

void PortableLockedShm::lock()
{
    (void) WaitForSingleObject(hMutex, INFINITE);
}

void PortableLockedShm::unlock()
{
    (void) ReleaseMutex(hMutex);
}

#else
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
PortableLockedShm::PortableLockedShm(const char *shmName, const char *mutexName, int mapSize) : size(mapSize)
{
    std::string filename;
    char shm_filename[128];
    filename.append("/");
    filename.append(shmName);
    //(void) shm_unlink(shm_filename);

    fd = shm_open(filename.c_str(), O_RDWR | O_CREAT, 0600);
    if (ftruncate(fd, mapSize) == 0) { ;; }
    else {
	fprintf(stderr, "oh, bother, ftruncate: %s\n", strerror(errno));
        //mem = (void*) -1;
    }
    mem = mmap(NULL, mapSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)0);
    if (mem == (void*)-1) {
	fprintf(stderr, "oh, bother, mmap: %s\n", strerror(errno));
    }
}

PortableLockedShm::~PortableLockedShm()
{
    //(void) shm_unlink(shm_filename);

    (void) munmap(mem, size);
    (void) close(fd);
}

void PortableLockedShm::lock()
{
    flock(fd, LOCK_EX);
}

void PortableLockedShm::unlock()
{
    flock(fd, LOCK_UN);
}



#endif
