#ifndef _SYSIO_H_
#define _SYSIO_H_
// this file contian utility for work with system io like ioctl

#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <string.h>

#include <iostream>

#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

typedef void * (*THREADFUNCPTR)(void *);

int xioctl(int fh, int request, void *arg);

void create_rt_pthread(pthread_t *ptid, THREADFUNCPTR func, void* args);

bool lock_thread_to_cpu(unsigned int cpuID, pthread_t* thread_ptr);

#endif