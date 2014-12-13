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
            230,    230,    16384,  7,      10,     16384,  461,    461,
            609,    609,    16384,  26,     30,     16384,  729,    729,
            831,    831,    16384,  46,     49,     16384,  922,    922,
            1004,   1004,   16384,  65,     69,     16384,  1081,   1081,
            1152,   1152,   16384,  85,     88,     16384,  1219,   1219,
            1283,   1283,   16384,  104,    107,    16384,  1343,   1343,
            1401,   1401,   16384,  124,    127,    16384,  1457,   1457,
            1511,   1511,   16384,  143,    146,    16384,  1563,   1563,
            1613,   1613,   16384,  163,    166,    16384,  1661,   1661,
            1708,   1708,   16384,  182,    185,    16384,  1754,   1754,
            1799,   1799,   16384,  201,    205,    16384,  1843,   1843,
            1885,   1885,   16384,  221,    224,    16384,  1927,   1927,
            1968,   1968,   16384,  240,    243,    16384,  2008,   2008,
            2047,   2047,   16384,  260,    263,    16384,  2085,   2085,
            2123,   2123,   16384,  279,    282,    16384,  2160,   2160,
            2196,   2196,   16384,  298,    301,    16384,  2231,   2231,
            2265,   2265,   16384,  317,    320,    16384,  2299,   2299,
            16384,  17,     20,     16384,  16384,  36,     39,     16384,
            16384,  56,     59,     16384,  16384,  75,     78,     16384,
            16384,  95,     98,     16384,  16384,  114,    117,    16384,
            16384,  133,    137,    16384,  16384,  153,    156,    16384,
            16384,  172,    176,    16384,  16384,  192,    195,    16384,
            16384,  211,    214,    16384,  16384,  230,    234,    16384,
            16384,  250,    253,    16384,  16384,  269,    272,    16384,
            16384,  289,    292,    16384,  16384,  308,    311,    16384,
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
