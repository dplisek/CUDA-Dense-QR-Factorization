// =============================================================================
// === GPUQREngine/Include/Kernel/sharedMemory.cu ==============================
// =============================================================================

#ifndef GPUQRENGINE_SHAREDMEMORY_HPP
#define GPUQRENGINE_SHAREDMEMORY_HPP

typedef union sharedMemory
{
    struct
    {
        #define MAX_MCHUNK 12

        // shared memory for factorize kernel (tile case)
        // size of A must match size of V in apply, below.
    	uint64_t A [PANELSIZE * TILESIZE + 1][TILESIZE + PADDING] ;
    	uint64_t T [TILESIZE + 1][TILESIZE + PADDING] ;
    	uint64_t Z [MAX_MCHUNK][TILESIZE+1] ;
    	uint64_t A1 [TILESIZE] ;
    	uint64_t V1 [TILESIZE] ;
    	uint64_t tau ;

        #undef MAX_MCHUNK
    } factorize ;

//    struct
//    {
//        #define MAX_COL_TILES 2
//
//        // shared memory for block_apply kernels
//        // size of V must match size of A in factorize, above.
//        double V [PANELSIZE * TILESIZE + 1][TILESIZE + PADDING] ;
//        double C [TILESIZE][MAX_COL_TILES * TILESIZE + PADDING] ;
//
//        #undef MAX_COL_TILES
//    } apply ;
//
//    struct
//    {
//        int Rimap[PACKASSEMBLY_SHMEM_MAPINTS];
//        int Rjmap[PACKASSEMBLY_SHMEM_MAPINTS];
//    } packassemble ;

} SharedMemory ;

/* Shared memory for all kernels is defined globally and allocated here. */
__shared__ SharedMemory shMemory;
__shared__ TaskDescriptor myTask;
//__shared__ int IsApplyFactorize;

#endif
