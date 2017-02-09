/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <mpi.h>

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;
static const int CPU_THREADS = 16;

unsigned char* GPU_Init(const int size);
void GPU_Exec(const int from_frame, const int to_frame, const int width, \
                unsigned char pic_d[]);
void GPU_Fini(const int size, unsigned char pic[], unsigned char pic_d[]);

int main(int argc, char *argv[])
{    
    const int NODE_ZERO = 0;
    int comm_sz; // # of MPI processes
    int my_rank; // Process order (0, 1, 2 ..)

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == NODE_ZERO) {
        printf("Fractal v1.5 [Hybrid1]\n");
        printf("Total number of processes == %d\n", comm_sz);
    }
    // check command line
    if (argc != 4) {
        fprintf(stderr, "usage: %s frame_width cpu_frames gpu_frames\n", \
            argv[0]); 
        exit(-1);
    }
    int width = atoi(argv[1]);
    if (width < 10) {
        fprintf(stderr, "error: frame_width must be at least 10\n"); 
        exit(-1);
    }
    int cpu_frames = atoi(argv[2]);
    if (cpu_frames < 0) {
        fprintf(stderr, "error: cpu_frames must be at least 0\n"); 
        exit(-1);
    }
    int gpu_frames = atoi(argv[3]);
    if (gpu_frames < 0) {
        fprintf(stderr, "error: gpu_frames must be at least 0\n"); 
        exit(-1);
    }
    int frames = cpu_frames + gpu_frames;
    if (frames < 1) {
        fprintf(stderr, \
            "error: total number of frames must be at least 1\n"); 
        exit(-1);
    }
    int total_frames = frames * comm_sz,
        total_cpu = cpu_frames * comm_sz,
        total_gpu = gpu_frames * comm_sz;
    if (my_rank == NODE_ZERO) {
        printf("computing %d frames of %d by %d fractal (%d CPU frames \
and %d GPU frames)\n", total_frames, width, width, \
        total_cpu, total_gpu);
    }
    const int from_frame = my_rank * frames,
              mid_frame = from_frame + gpu_frames,
              to_frame = mid_frame + cpu_frames,
              LOCAL_ALLOC = frames * width * width,
              GLOBAL_ALLOC = frames * width * width * comm_sz;
    // allocate picture arrays
    // pic will hold each node's calculated GPU + CPU frames
    unsigned char* pic = new unsigned char[LOCAL_ALLOC];
    unsigned char* master;
    if (my_rank == NODE_ZERO) {
        // master, allocated on NODE_ZERO only, will hold
        // GPU + CPU calculated frames for ALL nodes
        master = new unsigned char[GLOBAL_ALLOC];
    }
    // Only initialize CUDA subsystem if there are frames to be calculated
    // on the GPU
    unsigned char* pic_d = NULL;
    if (gpu_frames != 0) {
        pic_d = GPU_Init(gpu_frames * width * width * \
                  sizeof(unsigned char));
    }

    // start time
    struct timeval start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&start, NULL);

    // Asynchronously compute frames on GPU if there are frames to be
    // calculated. Additionally avoids initializing subsystem in the timed
    // code if gpu_frames == 0
    if (gpu_frames != 0) {
        GPU_Exec(from_frame, mid_frame, width, pic_d);
    } 

    // If there are frames to be calculated on the CPU
    if (cpu_frames != 0) {
        int frame,
            col,
            depth,
            row;
        double my_delta = Delta,
               x,
               x2,
               y,
               y2;
       
        double delta = Delta;
        #pragma omp parallel for num_threads(CPU_THREADS) default(none) \
         private(frame, my_delta, depth, row, x, x2, col, y, y2) \
         shared(pic, frames, width) \
         schedule(static, 1)
        for (int frame = mid_frame; frame < to_frame; ++frame) {
            my_delta = Delta * pow(0.99, frame + 1);
            const double xMin = xMid - my_delta;
            const double yMin = yMid - my_delta;
            const double dw = 2.0 * my_delta / width;

            for (int row = 0; row < width; ++row) {
                const double cy = -yMin - row * dw;
                for (int col = 0; col < width; ++col) {
                    const double cx = -xMin - col * dw;
                    double x = cx;
                    double y = cy;
                    int depth = 256;
                    double x2,
                           y2;
                    do {
                        x2 = x * x;
                        y2 = y * y;
                        y = 2 *x * y + cy;
                        x = x2 - y2 + cx;
                        --depth;
                    } while((depth > 0) && ((x2 + y2) < 5.0));
                    pic[(frame - from_frame) * width * width + \
                        row * width + col] = (unsigned char) depth;
                }
            }
        }
    }

    // the following call copies the GPU's result into the beginning
    // of the CPU's pic array
    if (gpu_frames != 0) {
        GPU_Fini(gpu_frames * width * width * sizeof(unsigned char), \
                    pic, pic_d);
    }
    MPI_Gather(pic, LOCAL_ALLOC, MPI_UNSIGNED_CHAR, \
                master, LOCAL_ALLOC, MPI_UNSIGNED_CHAR, NODE_ZERO, \
                    MPI_COMM_WORLD);
    // end time
    gettimeofday(&end, NULL);
    if (my_rank == NODE_ZERO) {
        double runtime = end.tv_sec + end.tv_usec / 1000000.0 - \
                         start.tv_sec - start.tv_usec / 1000000.0;
        printf("compute time: %.4f s\n", runtime);
    }
    // verify result by writing frames to BMP files
    if ((width <= 400) && (frames <= 30) && (my_rank == NODE_ZERO)) {
        for (int frame = 0; frame < total_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 10000);
            writeBMP(width, width, &master[frame * width * width], name);
        }
    }
    if (my_rank == NODE_ZERO) {
        delete [] master;
    }
    delete [] pic;

    MPI_Finalize();
    return 0;
}
