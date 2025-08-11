#include <cuda_runtime.h>
#include <cstdio>
extern "C" {

void* my_alloc(size_t sz, void* stream) {
    void* p;
    cudaMalloc(&p, sz);
    printf("[pool] alloc %zu\n", sz);
    return p;
}

void  my_free(void* p, size_t, void* stream) {
    cudaFree(p);
    printf("[pool] free\n");
}

} // extern "C"
