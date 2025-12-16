	/*
  Ecuaci�n del calor 2D con FFT usando Intel oneMKL (DFTI) + OpenMP

  - FFT 2D con MKL DFTI (in-place)
  - No recibe par�metros por consola
  - Usa OMP_NUM_THREADS (omp_get_max_threads) y lo aplica a MKL
  - Guarda output_heat.txt (x y u)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "mkl.h"       // MKL + DFTI
#include "mkl_dfti.h"  // DFTI

#define NX 2048
#define NY 2048
#define LX 1.0
#define LY 1.0
#define ALPHA 0.01
#define T_FINAL 1.0

static inline MKL_Complex16 c_mul_real(MKL_Complex16 a, double s) {
    MKL_Complex16 r; r.real = a.real * s; r.imag = a.imag * s; return r;
}

int main(void) {
    const int nthreads = omp_get_max_threads();

    // Forzar MKL a usar los hilos disponibles por OMP_NUM_THREADS
    mkl_set_dynamic(0);
    mkl_set_num_threads(nthreads);

    printf("============================================\n");
    printf(" ECUACION DEL CALOR 2D - FFT con Intel oneMKL\n");
    printf("============================================\n");
    printf("Nx = %d, Ny = %d\n", NX, NY);
    printf("Lx = %g, Ly = %g\n", LX, LY);
    printf("alpha = %g, t_final = %g\n", ALPHA, T_FINAL);
    printf("OpenMP threads (OMP_NUM_THREADS) = %d\n", nthreads);
    printf("MKL threads set to               = %d\n", mkl_get_max_threads());

    const double dx = LX / NX;
    const double dy = LY / NY;

    // Arreglo complejo 2D en memoria contigua (row-major): u[i*NY + j]
    MKL_Complex16 *u = (MKL_Complex16*) mkl_malloc((size_t)NX * (size_t)NY * sizeof(MKL_Complex16), 64);
    if (!u) {
        fprintf(stderr, "Error: no se pudo asignar memoria.\n");
        return 1;
    }

    // Condici�n inicial: gaussiana 2D centrada
    const double sigma = 0.05;
    printf("Condici�n inicial: u(x,y,0)=exp(-((x-Lx/2)^2+(y-Ly/2)^2)/sigma^2), sigma=%g\n", sigma);

    // Region paralela unicamente para definir la malla, no esta considerada como la solucion en si por lo que no cuenta en el tiempo
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            const double x = i * dx;
            const double y = j * dy;
            const double r2 = (x - LX/2.0)*(x - LX/2.0) + (y - LY/2.0)*(y - LY/2.0);
            const double val = exp(-r2 / (sigma * sigma));
            u[(size_t)i * NY + j].real = val;
            u[(size_t)i * NY + j].imag = 0.0;
        }
    }

    // Crear descriptor FFT 2D compleja
    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG status = 0;
    MKL_LONG lengths[2] = { NX, NY };

    status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 2, lengths);
    if (status != 0) {
        fprintf(stderr, "DFTI error CreateDescriptor: %s\n", DftiErrorMessage(status));
        mkl_free(u);
        return 1;
    }

    status = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status != 0) {
        fprintf(stderr, "DFTI error SetValue(PLACEMENT): %s\n", DftiErrorMessage(status));
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;

    const double invN = 1.0 / ((double)NX * (double)NY);
    status = DftiSetValue(desc, DFTI_BACKWARD_SCALE, invN);
    if (status != 0) {
        fprintf(stderr, "DFTI error SetValue(BACKWARD_SCALE): %s\n", DftiErrorMessage(status));
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;
    }

    status = DftiCommitDescriptor(desc);
    if (status != 0) {
        fprintf(stderr, "DFTI error CommitDescriptor: %s\n", DftiErrorMessage(status));
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;
    }

    // Medici�n de tiempo
    const double t0 = omp_get_wtime();

    // FFT directa
    status = DftiComputeForward(desc, u);
    if (status != 0) {
        fprintf(stderr, "DFTI error ComputeForward: %s\n", DftiErrorMessage(status));
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;
    }

    // Evoluci�n espectral (malla de frecuencias estilo FFT)
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < NX; p++) {
        for (int q = 0; q < NY; q++) {
            const int kx = (p <= NX/2) ? p : p - NX;
            const int ky = (q <= NY/2) ? q : q - NY;

            // k^2 = (kx/Lx)^2 + (ky/Ly)^2
            const double k2 = ( (double)kx / LX ) * ( (double)kx / LX )
                            + ( (double)ky / LY ) * ( (double)ky / LY );

            const double factor = exp(-4.0 * M_PI * M_PI * ALPHA * k2 * T_FINAL);
            u[(size_t)p * NY + q] = c_mul_real(u[(size_t)p * NY + q], factor);
        }
    }

    // IFFT (ya normalizada por BACKWARD_SCALE)
    status = DftiComputeBackward(desc, u);
    if (status != 0) {
        fprintf(stderr, "DFTI error ComputeBackward: %s\n", DftiErrorMessage(status));
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;
    }

    const double t1 = omp_get_wtime();
    printf("Tiempo total de ejecucion: %f segundos\n", t1 - t0);

    // Guardar resultados
    /*
    FILE *f = fopen("output_heat_mkl.txt", "w");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir output_heat_mkl.txt\n");
        DftiFreeDescriptor(&desc);
        mkl_free(u);
        return 1;
    }

    fprintf(f, "# Solucion ecuacion del calor 2D (MKL)\n");
    fprintf(f, "# Nx=%d Ny=%d Lx=%g Ly=%g alpha=%g t=%g\n", NX, NY, LX, LY, ALPHA, T_FINAL);
    fprintf(f, "# Columnas: x y u(x,y,t)\n\n");

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            const double x = i * dx;
            const double y = j * dy;
            const double val = u[(size_t)i * NY + j].real; // parte real
            fprintf(f, "%.15f %.15f %.15e\n", x, y, val);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    printf("Resultados guardados en output_heat_mkl.txt\n");
    */
    // Limpieza
    DftiFreeDescriptor(&desc);
    mkl_free(u);

    // ------------ GUARDAR EN ARCHIVO ------------------------
    FILE *file = fopen("Tiemplo_MKL.txt", "a");
    if (file == NULL) {
        printf("Error al abrir el archivo de salida.\n");
        return 1;
    }

    #ifdef _OPENMP
    //fprintf(file, "%d\t%f\t%f\t%f\t%g\t%d\n", N, min_x, min_y, min_f, time, num_threads);
    fprintf(file, "%d\t%f\n", nthreads,  t1 - t0);
    #endif

    fclose(file);

    return 0;
}
