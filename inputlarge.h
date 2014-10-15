#include <stdio.h>
#include <limits.h>
#include <errno.h>

// Simple Linear Congruential Random Number Generator
// to ensure the input data is the same on all platforms.
struct LCG
{
    LCG(unsigned int seed_, unsigned int maxval_)
    {
        seed = seed_;
        maxval = maxval_;
    }
    unsigned int get()
    {
        seed = seed * 1664525U + 1013904223U;
        return (unsigned int)(maxval * ((float)seed / (float)UINT_MAX));
    }

    unsigned int seed;
    unsigned int maxval;
};

unsigned int *get_input(void)
{
        unsigned int *data, *ptr;
        int num_maps, rows, columns, i, j, datasize;
        // Row-column table to generate 100 maps with large input data.
        // The table is randomly generated
        int rctable[] =
        {
            16384,16384,16384,16384,16384,16384
        };
        num_maps = (sizeof(rctable) / sizeof(int)) / 2;
        datasize = 1; // + 1 For num_maps
        for(i = 0; i < num_maps; ++i) {
            rows = rctable[2 * i];
            columns = rctable[2 * i + 1];
            datasize += 2 + rows * columns; // #rows, #columns, data
        }

        data = (unsigned int *)malloc(datasize * sizeof(int));
        if(!data) {
            printf("Insufficient memory %d\n", errno);
            exit(-1);
        }

        LCG lcg(123, 65535); 
        ptr = data;
        *ptr++ = num_maps;
        for(i = 0; i < num_maps; i++) {
            *ptr++ = rows = rctable[2 * i];
            *ptr++ = columns = rctable[2 * i + 1];
            for(j = 0; j < rows*columns; j++) {
                if(j%256) {
                    *ptr++ = 0;
                    continue;
                }
                *ptr++ = lcg.get();
            }
        }
    return data;
}

#define NUM_ITERATIONS 10
