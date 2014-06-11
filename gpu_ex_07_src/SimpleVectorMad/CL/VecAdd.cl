// Device code
kernel void VecAdd(global float* A, global float* B, global float* C, global float* D, global float* E)
{
    int i = get_global_id(0);
    //printf("    [GPU] globalID: %d\n",i);
    C[i] = A[i] + B[i];
    E[i] = A[i] + B[i] * D[i];
}
