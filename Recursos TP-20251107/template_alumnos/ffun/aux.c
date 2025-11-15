__attribute__((visibility("default"))) void calcularAX(float* A, float* x, float* result, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i] += A[i * m + j] * x[j];
        }
    }
}

__attribute__((visibility("default"))) void matMul(float* A, float* B, float* result, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                result[i * p + j] += A[i * m + k] * B[k * p + j];
            }
        }
    }
}