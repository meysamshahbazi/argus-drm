
#include "sysio.h"
#include <sys/mman.h>
#include <malloc.h>

int xioctl(int fh, int request, void *arg)
{
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && ( EINTR == errno /* ||  EBUSY == errno */) );

    return r;
}

static int PREALLOC_SIZE     = 200 * 1024 * 1024;

void create_rt_pthread(pthread_t *ptid, THREADFUNCPTR func, void* args)
{
    // https://wiki.linuxfoundation.org/realtime/documentation/howto/applications/application_base
    struct sched_param sch_param;
    pthread_attr_t attr;
    int ret;
    // /* Lock memory */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        printf("mlockall failed: %m\n");
    }

    // -------------------------------------------------------------------------
    // turn off malloc trimming.
	mallopt(M_TRIM_THRESHOLD, -1);

	// turn off mmap usage.
	mallopt(M_MMAP_MAX, 0);

	unsigned int page_size = sysconf(_SC_PAGESIZE);
	unsigned char * buffer = (unsigned char *)malloc(PREALLOC_SIZE);

	// touch each page in this piece of memory to get it mapped into RAM
	for(int i = 0; i < PREALLOC_SIZE; i += page_size)
	{
		// each write to this buffer will generate a pagefault.
		// once the pagefault is handled a page will be locked in memory and never
		// given back to the system.
		buffer[i] = 0;
	}
		
	// release the buffer. As glibc is configured such that it never gives back memory to
	// the kernel, the memory allocated above is locked for this process. All malloc() and new()
	// calls come from the memory pool reserved and locked above. Issuing free() and delete()
	// does NOT make this locking undone. So, with this locking mechanism we can build applications
	// that will never run into a major/minor pagefault, even with swapping enabled.
	free(buffer);
    // -------------------------------------------------------------------------

    /* Initialize pthread attributes (default values) */
    ret = pthread_attr_init(&attr);
    if (ret) {
        printf("init pthread attributes failed\n");
    }

    /* Set a specific stack size  */
    ret = pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
    if (ret) {
        printf("pthread setstacksize failed\n");
    }

    /* Set scheduler policy and priority of pthread */
    ret = pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    if (ret) {
        printf("pthread setschedpolicy failed\n");
    }
    sch_param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    ret = pthread_attr_setschedparam(&attr, &sch_param);
    if (ret) {
        printf("pthread setschedparam failed\n");

    }
    /* Use scheduling parameters of attr */
    ret = pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    if (ret) {
        printf("pthread setinheritsched failed\n");
    }

    pthread_create(ptid, &attr, func, args); 
}

bool lock_thread_to_cpu(unsigned int cpuID, pthread_t* thread_ptr )
{
	// pthread_t thread;
	// cpu_set_t cpu_set;

	// CPU_ZERO(&cpu_set);
	// CPU_SET(cpuID, &cpu_set);

	// // if( !thread_ptr )
	// // 	thread = pthread_self();
	// // else

    // thread = *thread_ptr;

	// const int result_set_cpu = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpu_set);

	// if( result_set_cpu != 0 )
	// {
	// 	printf("pthread_setaffinity_np() failed (error=%i)\n", result_set_cpu);
	// 	return false;
	// }

    // /*
	// const int result_get_cpu = pthread_getaffinity_np(thread_self, sizeof(cpu_set_t), &cpu_set);

	// if( result_get_cpu != 0 )
	// {
	// 	printf("pthread_getaffinity_np() failed (error=%i)\n", result_get_cpu);
	// 	return false;
	// }
    // */
    
	// const int cpu_set_count = CPU_COUNT(&cpu_set);
	// printf("cpu_set_count=%i\n", cpu_set_count);

	return true;
}

