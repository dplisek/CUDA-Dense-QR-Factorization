/*
 ============================================================================
 Name        : DenseQR3by1.cu
 Author      : Tim Davis et al
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"
#include "sharedMemory.hpp"

#define FACTORIZE       factorize_3_by_1_tile_vt
#define ROW_PANELSIZE   (3)
#define M               (ROW_PANELSIZE * TILESIZE)
#define N               (TILESIZE)
#define BITTYROWS       (8)
#define MODULE_MAX_BITS	(32) // to allow squaring withing uint64_t

#define INV(n, module)	pow_mod(n, module - 2, module)

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit( EXIT_FAILURE);
	}
}

__device__ __host__ uint64_t pow_mod(uint64_t x, uint64_t b, uint64_t module) {
	uint64_t z = 1;
	uint64_t i = ((uint64_t) 1) << MODULE_MAX_BITS - 1; // has only a 1 in the highest possible bit in module

	while (i && !(b & i)) {
		i = i >> 1;
	}

	while (i) {
		z = (z*z) % module;
		if (b & i) z = (z*x) % module;
		i = i >> 1;
	}

	return z;
}

__device__ __host__ uint64_t sqrt_mod(uint64_t n, uint64_t module, uint64_t sqrtQ, uint64_t sqrtS, uint64_t sqrtZ, uint64_t *nLeg) {
	uint64_t sqrt, test, prevRequiredSquares, requiredSquares, c, iterator, squarer;

	// test quadratic residue
	*nLeg = pow_mod(n, (module - 1) / 2, module);
	if (*nLeg != 1) return n;

	// initialize loop variables
	prevRequiredSquares = sqrtS;
	c = sqrtZ;
	sqrt = pow_mod(n, ((sqrtQ + 1) / 2), module);
	test = pow_mod(n, sqrtQ, module);

	// sqrt generation loop
	while (test != 1) {

		// c squaring loop
		squarer = test;
		requiredSquares = 1;
		bool doneSquaring = false;
		for (iterator = 0; iterator < prevRequiredSquares - 2; ++iterator) {
			if (doneSquaring || (squarer = (squarer*squarer) % module) == 1) {
				doneSquaring = true;
				c = (c*c) % module;
			} else {
				++requiredSquares;
			}
		}
		prevRequiredSquares = requiredSquares;

		// update loop variables
		sqrt = (sqrt*c) % module;
		c = (c*c) % module;
		test = (test*c) % module;

	}

	return sqrt;
}

__device__ void FACTORIZE ( )
{

    //--------------------------------------------------------------------------
    // The bitty block
    //--------------------------------------------------------------------------

    // The matrix A is M by N, T is N by N (if present)

    // Each thread owns an r-by-1 bitty column of A.  If its first entry is
    // A(ia,ja), and then it owns entries in every mchunk rows after that
    // [A(ia,ja), A(ia+mchunk,ja), ... ].
    // Each column is operated on by mchunk threads.

    #define MCHUNK          (M / BITTYROWS)
    #define MYBITTYROW(ii)  (ii*MCHUNK + (threadIdx.x % MCHUNK))
    #define MYBITTYCOL      (threadIdx.x / MCHUNK)
    #define ATHREADS        (MCHUNK * N)
    #define WORKER          (ATHREADS == NUMTHREADS || threadIdx.x < ATHREADS)

    uint64_t rbitA [BITTYROWS] ;     // bitty block for A
    uint64_t rbitV [BITTYROWS] ;     // bitty block for V
    uint64_t module ;
    uint64_t sigma ;                 // only used by thread zero
    uint64_t sqrtQ, sqrtS, sqrtZ ;	 // only used by thread zero for square roots

    //--------------------------------------------------------------------------
    // shared memory usage
    //--------------------------------------------------------------------------

    #define shA         shMemory.factorize.A
    #define shZ         shMemory.factorize.Z
    #define shRdiag     shMemory.factorize.A1
    #define RSIGMA(i)   shMemory.factorize.V1 [i]
	#define shModule	shMemory.factorize.module

//    #ifdef WHOLE_FRONT
//        // T is not computed, and there is no list of tiles
//        #define TAU     shMemory.factorize.tau
//    #else
        // T is computed and saved in the VT tile, work on a set of tiles
        #define TAU     shT [k][k]
        #define shT     shMemory.factorize.T
        #define shV1    shMemory.factorize.V1
//    #endif


    //--------------------------------------------------------------------------
    // Grab my task from the queue.
    //--------------------------------------------------------------------------

    int fn = myTask.fn ;

//    #ifdef WHOLE_FRONT
//
//        int fm = myTask.fm ;
//        int nv = MIN (fm, fn) ;
//        // If nv is a constant, it allows the outer loop to be unrolled:
//        // #define nv N
//        #define j1 0
//        // The whole front always considers the edge case
//        #ifndef EDGE_CASE
//        #define EDGE_CASE
//        #endif
//
//    #else

        int j1 = myTask.extra[4] ;
//        #ifdef EDGE_CASE
//        int fm = myTask.fm ;
//        int nv = MIN (fm, fn) - j1 ;
//        nv = MIN (nv, N) ;
//        nv = MIN (nv, M) ;
//        #else
        #define nv N
//        #endif
        uint64_t (*glVT)[TILESIZE] = (uint64_t (*)[TILESIZE]) myTask.AuxAddress[0] ;

//    #endif

//    #ifdef EDGE_CASE
//        // Check if an entry is inside the front.
//        #define INSIDE(test) (test)
//    #else
        // The entry is guaranteed to reside inside the frontal matrix.
//        #define INSIDE(test) (1)
//    #endif

    // bool is_false = (fn < 0) ;

    #define glA(i,j)        (myTask.F[((i)*fn + (j))])

    //--------------------------------------------------------------------------
    // Load A into shared memory
    //--------------------------------------------------------------------------

    // ACHUNKSIZE must be an integer
    #define it              (threadIdx.x / N)
    #define jt              (threadIdx.x % N)
    #define ACHUNKSIZE      (NUMTHREADS / N)

//    #ifdef WHOLE_FRONT
//
//        // all threads load the entire front (no tiles).
//        // always check the edge case.
//        // M / ACHUNKSIZE must be an integer.
//        #define NACHUNKS    (M / ACHUNKSIZE)
//        for (int ii = 0 ; ii < NACHUNKS ; ii++)
//        {
//            int i = ii * ACHUNKSIZE + it ;
//            shA [i][jt] = (i < fm && jt < fn) ?  glA (i, jt) : 0 ;
//        }
//
//    #else

        // when all threads work on a tile.
        // (N*N / NUMTHREADS) does not have to be an integer.  With a tile
        // size of N=32, and NUMTHREADS=384, it isn't.  So compute the ceiling,
        // and handle the clean up by testing i < N below.
        #define NACHUNKS    CEIL (N*N, NUMTHREADS)

        /* If we're not coming from an apply-factorize, load from F. */
//        if(IsApplyFactorize == 0)
//        {
            // Load tiles from the frontal matrix
            // accounts for 25% of the total time on Kepler, 13% on Fermi
            for (int t = 0 ; t < ROW_PANELSIZE ; t++)
            {
                int rowTile = myTask.extra[t];
//                if (INSIDE (rowTile != EMPTY))
//                {
                    /* load the tile of A from global memory */
                    for (int ii = 0 ; ii < NACHUNKS ; ii++)
                    {
                        int i = ii * ACHUNKSIZE + it ;
                        if (ii < NACHUNKS-1 || i < N)
                        {
                            shA [i + t*TILESIZE][jt] =
//                              (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn)) ?
                              glA (i+rowTile, jt+j1) ; // : 0 ;
                        }
                    }
//                }
//                else
//                {
//                    /* clear the tile of A */
//                    for (int ii = 0 ; ii < NACHUNKS ; ii++)
//                    {
//                        int i = ii * ACHUNKSIZE + it ;
//                        if (ii < NACHUNKS-1 || i < N)
//                        {
//                            shA [i + t*TILESIZE][jt] = 0 ;
//                        }
//                    }
//                }
            }
//        }

        // clear the tile T
        for (int ii = 0 ; ii < NACHUNKS ; ii++)
        {
            int i = ii * ACHUNKSIZE + it ;
            if (ii < NACHUNKS-1 || i < N)
            {
                shT [i][jt] = 0 ;
            }
        }
//    #endif



    // Load the module from global memory to shared
    if (threadIdx.x == 0) {
    	shModule = *(myTask.AuxAddress[1]);
    }

    /* We need all of A to be loaded and T to be cleared and module to be ready before proceeding. */
    __syncthreads();

    // Load the module from shared memory to register
    module = shModule;


    //--------------------------------------------------------------------------
    // precompute square root common parameters q, s, z for thread 0,
    // which is going to be the only one performing square roots
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0) {
    	sqrtQ = module - 1;
		sqrtS = 0;
		while (!(sqrtQ & 1)) {
			sqrtQ = sqrtQ >> 1;
			sqrtS += 1;
		}
		sqrtZ = 2;
		while (pow_mod(sqrtZ, (module - 1) / 2, module) != module - 1) ++sqrtZ;
		sqrtZ = pow_mod(sqrtZ, sqrtQ, module);
    }


    //--------------------------------------------------------------------------
    // load A into the bitty block
    //--------------------------------------------------------------------------

    if (WORKER)
    {
        #pragma unroll
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            rbitA [ii] = shA [i][MYBITTYCOL] ;
        }
    }

    //--------------------------------------------------------------------------
    // compute the first sigma = sum (A (1:m,1).^2)
    //--------------------------------------------------------------------------

    if (WORKER && MYBITTYCOL == 0)
    {
        // each thread that owns column 0 computes sigma for its
        // own bitty block
        uint64_t s = 0 ;
        #pragma unroll
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            if (i >= 1)
            {
                s += (rbitA [ii] * rbitA [ii]) % module;
                s %= module;
            }
        }
        RSIGMA (threadIdx.x) = s ;
    }

    // thread zero must wait for RSIGMA
    __syncthreads ( ) ;

    if (threadIdx.x == 0)
    {
        sigma = 0 ;
        #pragma unroll
        for (int ii = 0 ; ii < MCHUNK ; ii++)
        {
            sigma = (sigma + RSIGMA (ii)) % module ;
        }
        //printf("Total sigma for column 0: %llu\n", sigma);
    }

    //--------------------------------------------------------------------------
    // Do the block householder factorization
    //--------------------------------------------------------------------------

    // loop unrolling has no effect on the edge case (it is not unrolled),
    // but greatly speeds up the non-edge case.
    #pragma unroll
    for (int k = 0 ; k < nv ; k++)
    {

        //----------------------------------------------------------------------
        // write the kth column of A back into shared memory
        //----------------------------------------------------------------------

        if (WORKER && MYBITTYCOL == k && k > 0)
        {
            // the bitty block for threads that own column k contains
            // the kth column of R and the kth column of v.
            #pragma unroll
            for (int ii = 0 ; ii < BITTYROWS ; ii++)
            {
                int i = MYBITTYROW (ii) ;
                shA [i][k] = rbitA [ii] ;
            }
        }

        __syncthreads ( ) ;

        // At this point, A (:,k) is held in both shared memory, and in the
        // threads that own that column.  A (:,k) is not yet the kth
        // Householder vector, except for the diagnal (which is computed
        // below).  A (0:k-1,k) is now the kth column of R (above the
        // diagonal).

        //----------------------------------------------------------------------
        // compute the Householder coefficients
        //----------------------------------------------------------------------

        // This is costly, accounting for about 25% of the total time on
        // Kepler, and 22% on Fermi, when A is loaded from global memory.  This
        // means the work here is even a higher fraction when A is in shared.
        if (threadIdx.x == 0)
        {
            uint64_t x1 = shA [k][k] ;            // the diagonal A (k,k)
            uint64_t s, v1, tau, sLeg ;

//            if (sigma <= EPSILON)
            if (sigma == 0)
            {
                printf ("Error in column %d: Hit sigma = 0, v1 would become 0, cannot invert 0 to get tau. Exiting.\n", k) ;
                return;
//                s = x1 ;
//                v1 = 0 ;
//                tau = 0 ;
            }
            else
            {
            	s = (((x1*x1) % module) + sigma) % module ;
            	if (s == 0) {
                    printf ("Error in column %d: Hit s = 0, cannot invert 0 to get tau. Exiting.\n", k) ;
                    return;
            	}
                s = sqrt_mod (s, module, sqrtQ, sqrtS, sqrtZ, &sLeg) ;
                if (sLeg == module - 1) {
                	printf ("Error in column %d: Cannot determine size of vector, no square root of s = %llu. Exiting.\n", k, s);
                	return;
                }
                v1 = (module + x1 - s) % module ; // prevent unsigned underflow by prepending a module
                tau = module - INV( (s * v1) % module , module ) ; // prevent unsigned underflow by prepending a module
                //printf("Successfully computed Householder coefficients for column %d. s = %llu, v1 = %llu, tau = %llu.\n", k, s, v1, tau);
            }
            shRdiag [k] = s ;       // the diagonal entry of R
            shA [k][k] = v1 ;       // the topmost entry of the vector v
            TAU = tau ;             // tile case: T (k,k) = tau
        }

        // All threads need v1, and later on they need tau
        __syncthreads ( ) ;


        // A (0:k-1,k) now holds the kth column of R (excluding the diagonal).
        // A (k:m-1,k) holds the kth Householder vector (incl. the diagonal).

        //----------------------------------------------------------------------
        // z = (-tau) * v' * A (k:m-1,:), where v is A (k:m-1,k)
        //----------------------------------------------------------------------

        if (WORKER) // && (COMPUTE_T || MYBITTYCOL > k))
        {
            // Load the vector v from A (k:m-1,k) into the V bitty block.
            // If T is not computed and MYBITTYCOL <= k, then this code can
            // be skipped, but the code is slower when that test is made.
            #pragma unroll
            for (int ii = 0 ; ii < BITTYROWS ; ii++)
            {
                int i = MYBITTYROW (ii) ;
                // only i >= k is needed, but it's faster to load it all
                rbitV [ii] = shA [i][k] ;
            }

            // compute v' * A (k:m-1,:), each thread in its own column
            {
                uint64_t z = 0 ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k)
                    {
                        z += (rbitV [ii] * rbitA [ii]) % module;
						z %= module;
                    }
                }
                // store z into the reduction space in shared memory
                shZ [MYBITTYROW(0)][MYBITTYCOL] = z ;
            }
        }

        // All threads need to see the reduction space for Z
        __syncthreads ( ) ;

        // Reduce Z into a single row vector z, using the first warp only
        if (threadIdx.x < N) // && (COMPUTE_T || threadIdx.x > k))
        {
            uint64_t z = 0 ;
            #pragma unroll
            for (int ii = 0 ; ii < MCHUNK ; ii++)
            {
                z += shZ [ii][threadIdx.x] ;
                z %= module;
            }
            shZ [0][threadIdx.x] = module - ((z * TAU) % module) ;
        }

        // All threads need to see the z vector
        __syncthreads ( ) ;

        //----------------------------------------------------------------------
        // update A (in register) and compute the next sigma
        //----------------------------------------------------------------------

        if (WORKER && MYBITTYCOL > k)
        {
            // A (k:m,k+1:n) = A (k:,k+1:n) + v * z (k+1:n) ;
            // only threads that own a column MYBITTYCOL > k do any work
            {
                uint64_t z = shZ [0][MYBITTYCOL] ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k)
                    {
                        rbitA [ii] += (rbitV [ii] * z) % module ;
                        rbitA [ii] %= module;
                    }
                }
            }

            // sigma = sum (A ((k+2):m,k+1).^2), except for the reduction
            if (MYBITTYCOL == k+1)
            {
                // each thread that owns column k+1 computes sigma for its
                // own bitty block
                uint64_t s = 0 ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k+2)
                    {
                        s += (rbitA [ii] * rbitA [ii]) % module;
                        s %= module;
                    }
                }
                RSIGMA (MYBITTYROW(0)) = s ;
            }
        }

        //----------------------------------------------------------------------
        // construct the kth column of T
        //----------------------------------------------------------------------

//        #ifndef WHOLE_FRONT

            // T (0:k-1,k) = T (0:k-1,0:k-1) * z (0:k-1)'
            if (threadIdx.x < k)
            {
                uint64_t t_ik = 0 ;
                for (int jj = 0 ; jj < k ; jj++)
                {
                    t_ik += (shT [threadIdx.x][jj] * shZ [0][jj]) % module;
                    t_ik %= module;
                }
                shT [threadIdx.x][k] = t_ik ;
            }

//        #endif

        //----------------------------------------------------------------------
        // reduce sigma into a single scalar for the next iteration
        //----------------------------------------------------------------------

        // Thread zero must wait for RSIGMA
        __syncthreads ( ) ;

        if (threadIdx.x == 0)
        {
            sigma = 0 ;
            #pragma unroll
            for (int ii = 0 ; ii < MCHUNK ; ii++)
            {
                sigma = (sigma + RSIGMA (ii)) % module ;
            }
            //printf("Total sigma for column %d: %llu\n", k+1, sigma);
        }
    }

    // tril (A) now holds all the Householder vectors, including the diagonal.
    // triu (A,1) now holds R, excluding the diagonal.
    // shRdiag holds the diagonal of R.

    //--------------------------------------------------------------------------
    // write out the remaining columns of R, if any
    //--------------------------------------------------------------------------

    if (WORKER && MYBITTYCOL >= nv)
    {
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            shA [i][MYBITTYCOL] = rbitA [ii] ;
        }
    }

    //--------------------------------------------------------------------------

    /* Have a warp shuffle memory around. */
    if (threadIdx.x < N)
    {
//        #ifndef WHOLE_FRONT
        shV1 [threadIdx.x] = shA [threadIdx.x][threadIdx.x];
//        #endif
        shA [threadIdx.x][threadIdx.x] = shRdiag [threadIdx.x];
    }

    // Wait for the memory shuffle to finish before saving A to global memory
    __syncthreads();

    //--------------------------------------------------------------------------
    // save results back to global memory
    //--------------------------------------------------------------------------

//    #ifdef WHOLE_FRONT
//
//        if (jt < fn)
//        {
//            for (int ii = 0 ; ii < NACHUNKS ; ii++)
//            {
//                int i = ii * ACHUNKSIZE + it ;
//                if (i < fm) glA (i, jt) = shA [i][jt] ;
//            }
//        }
//
//    #else

        // Save VT back to global memory & clear out
        // lower-triangular part of the first tile (leaving R).
        for (int th=threadIdx.x; th<TILESIZE*TILESIZE; th+=blockDim.x)
        {
            int i = th / 32;
            int j = th % 32;

            /* The upper triangular part (including diagonal) is T. */
            if(i <= j)
            {
                glVT[i][j] = shT[i][j];
            }
            /* The lower triangular part is V.
             * Note we clear the tril part leaving only R in this tile. */
            else
            {
                glVT[i+1][j] = shA[i][j];
                shA[i][j] = 0;
            }
        }

        // Save the diagonal
        if (threadIdx.x < N)
        {
            glVT[threadIdx.x+1][threadIdx.x] = shV1[threadIdx.x];
        }

        // Wait for this operation to complete before saving A back to global
        // memory
        __syncthreads();

        // save the tiles in A back into the front in global memory
        for (int t = 0 ; t < ROW_PANELSIZE ; t++)
        {
            int rowTile = myTask.extra[t];
//            if (INSIDE (rowTile != EMPTY))
//            {
                for (int ii = 0 ; ii < NACHUNKS ; ii++)
                {
                    int i = ii * ACHUNKSIZE + it ;
                    if (ii < NACHUNKS-1 || i < N)
                    {
//                        if (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn))
//                        {
                            glA (i+rowTile, jt+j1) = shA [i + t*TILESIZE][jt];
//                        }
                    }
                }
//            }
        }
//    #endif
}



__global__ void qrKernel
(
    TaskDescriptor* Queue,
    int QueueLength
)
{
    /* Copy the task details to shared memory. */
    if(threadIdx.x == 0)
    {
//        IsApplyFactorize = 0;
        myTask = Queue[blockIdx.x];
    }
    __syncthreads();

    switch(myTask.Type)
    {
//        case TASKTYPE_SAssembly:    sassemble();    return;
//        case TASKTYPE_PackAssembly: packassemble(); return;

          case TASKTYPE_FactorizeVT_3x1:  factorize_3_by_1_tile_vt();      return;
//        case TASKTYPE_FactorizeVT_2x1:  factorize_2_by_1_tile_vt();      return;
//        case TASKTYPE_FactorizeVT_1x1:  factorize_1_by_1_tile_vt();      return;
//        case TASKTYPE_FactorizeVT_3x1e: factorize_3_by_1_tile_vt_edge(); return;
//        case TASKTYPE_FactorizeVT_2x1e: factorize_2_by_1_tile_vt_edge(); return;
//        case TASKTYPE_FactorizeVT_1x1e: factorize_1_by_1_tile_vt_edge(); return;
//        case TASKTYPE_FactorizeVT_3x1w: factorize_96_by_32();            return;
//
//        case TASKTYPE_Apply3: block_apply_3(); return;
//        case TASKTYPE_Apply2: block_apply_2(); return;
//        case TASKTYPE_Apply1: block_apply_1(); return;

//        #ifdef GPUQRENGINE_PIPELINING
//        // Apply3_Factorize[3 or 2]: (note fallthrough to next case)
//        case TASKTYPE_Apply3_Factorize3:
//            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 3;
//
//        case TASKTYPE_Apply3_Factorize2:
//            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 2;
//            block_apply_3_by_1();
//            break;
//
//        // Apply2_Factorize[3, 2, or 1]: (note fallthrough to next case)
//        case TASKTYPE_Apply2_Factorize3:
//            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 3;
//
//        case TASKTYPE_Apply2_Factorize2:
//            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 2;
//
//        case TASKTYPE_Apply2_Factorize1:
//            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 1;
//            block_apply_2_by_1();
//            break;
//        #endif

        default: break;
    }

//    #ifdef GPUQRENGINE_PIPELINING
//    /* Tasks that get to this point are Apply-Factorize tasks
//       because all other should have returned in the switch above. */
//    switch(myTask.Type)
//    {
//        case TASKTYPE_Apply3_Factorize3:
//        case TASKTYPE_Apply2_Factorize3: factorize_3_by_1_tile_vt_edge(); break;
//        case TASKTYPE_Apply3_Factorize2:
//        case TASKTYPE_Apply2_Factorize2: factorize_2_by_1_tile_vt_edge(); break;
//        case TASKTYPE_Apply2_Factorize1: factorize_1_by_1_tile_vt_edge(); break;
//    }
//    #endif
}

int main() {
	int dev;
	int numTasks = 10;
	cudaDeviceProp prop;
	uint64_t *F, *module;
	TaskDescriptor *queueHost, *queueDev;

	HANDLE_ERROR(cudaGetDevice(&dev));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));

	printf("Device: %s\n", prop.name);

	srand(15);

	HANDLE_ERROR(cudaMalloc((void **) &(queueDev), numTasks * sizeof(TaskDescriptor)));
	HANDLE_ERROR(cudaHostAlloc((void **) &(queueHost), numTasks * sizeof(TaskDescriptor), cudaHostAllocDefault));

	HANDLE_ERROR(cudaHostAlloc((void **) &(F), M * N * sizeof(uint64_t), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void **) &(module), sizeof(uint64_t), cudaHostAllocDefault));

	*module = 4294967291; // 32-bit

	for (int i = 0; i < numTasks; ++i) {

		queueHost[i].Type = TASKTYPE_FactorizeVT_3x1;
		queueHost[i].fm = M;
		queueHost[i].fn = N;
		if (ROW_PANELSIZE > 4) {
			printf("Cannot operate on ROW_PANELSIZE > 4.\n");
			return EXIT_FAILURE;
		}
		for (int j = 0; j < ROW_PANELSIZE; ++j) {
			queueHost[i].extra[j] = j * TILESIZE; // set the global memory row offsets
		}
		queueHost[i].extra[4] = 0;

		HANDLE_ERROR(cudaMalloc((void **) &(queueHost[i].F), M * N * sizeof(uint64_t))); // memory for input matrix
		HANDLE_ERROR(cudaMalloc((void **) &(queueHost[i].AuxAddress[0]), (N+1) * N * sizeof(uint64_t))); // memory for output V/T tile
		HANDLE_ERROR(cudaMalloc((void **) &(queueHost[i].AuxAddress[1]), sizeof(uint64_t))); // memory for module

		char filename[255];
		sprintf(filename, "task%03d", i);
		FILE *f = fopen(filename, "r");
		if (!f) {
			printf("Cannot open file %s for reading.\n", filename);
		}
		for (int m = 0; m < queueHost[i].fm; ++m) {
			for (int n = 0; n < queueHost[i].fn; ++n) {
				fscanf(f, "%llu", &(F[m*queueHost[i].fn + n]));
				//F[m*queueHost[i].fn + n] = rand() % *module;
				//printf("%010llu", F[m*queueHost[i].fn + n]);
				//if (n < queueHost[i].fn - 1) printf("\t");
			}
			//printf("\n");
		}
		fclose(f);

		HANDLE_ERROR(cudaMemcpy(queueHost[i].F, F, M * N * sizeof(uint64_t), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(queueHost[i].AuxAddress[1], module, sizeof(uint64_t), cudaMemcpyHostToDevice));
	}
	HANDLE_ERROR(cudaMemcpy(queueDev, queueHost, numTasks * sizeof(TaskDescriptor), cudaMemcpyHostToDevice));


	dim3 blocks(numTasks);
	dim3 threads(NUMTHREADS);

	qrKernel<<<blocks, threads>>>(queueDev, numTasks);

	cudaDeviceSynchronize();

	for (int i = 0; i < numTasks; ++i) {
//		HANDLE_ERROR(cudaMemcpy(F, queueHost[i].F, M * N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//
//		printf("\nTask %d R:\n", i);
//		for (int m = 0; m < queueHost[i].fm; ++m) {
//			for (int n = 0; n < queueHost[i].fn; ++n) {
//				printf("%010llu ", F[m*queueHost[i].fn + n]);
//			}
//			printf("\n");
//		}
//
//		HANDLE_ERROR(cudaMemcpy(F, queueHost[i].AuxAddress[0], (N+1) * N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
//
//		printf("\nTask %d VT:\n", i);
//		for (int m = 0; m < queueHost[i].fn + 1; ++m) {
//			for (int n = 0; n < queueHost[i].fn; ++n) {
//				printf("%010llu ", F[m*queueHost[i].fn + n]);
//			}
//			printf("\n");
//		}

		HANDLE_ERROR(cudaFree(queueHost[i].F));
		HANDLE_ERROR(cudaFree(queueHost[i].AuxAddress[0]));
		HANDLE_ERROR(cudaFree(queueHost[i].AuxAddress[1]));
	}

	HANDLE_ERROR(cudaFree(queueDev));
	HANDLE_ERROR(cudaFreeHost(queueHost));
	HANDLE_ERROR(cudaFreeHost(F));
	HANDLE_ERROR(cudaFreeHost(module));


	printf("Done.\n");
	return EXIT_SUCCESS;
}
