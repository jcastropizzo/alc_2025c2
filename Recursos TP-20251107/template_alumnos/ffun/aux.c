__attribute__((visibility("default"))) void calcularAX(float* A, float* x, float* result, int n, int m) {
    // A is n x m
    // x is m
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i] += A[i * m + j] * x[j];
        }
    }
}