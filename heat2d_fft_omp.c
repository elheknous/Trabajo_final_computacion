#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define NX 2048
#define NY 2048
#define LX 1.0
#define LY 1.0
#define ALPHA 0.01
#define T_FINAL 1.0

/* ===================== COMPLEJO ===================== */
typedef struct {
    double re;
    double im;
} complex_t;

complex_t c_add(complex_t a, complex_t b){ return (complex_t){a.re+b.re, a.im+b.im}; }
complex_t c_sub(complex_t a, complex_t b){ return (complex_t){a.re-b.re, a.im-b.im}; }
complex_t c_mul(complex_t a, complex_t b){
    return (complex_t){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}

/* ===================== FFT 1D ===================== */
void fft1d(complex_t *x, int n, int invert){
    for(int i=1,j=0;i<n;i++){
        int bit = n>>1;
        for(; j&bit; bit>>=1) j ^= bit;
        j |= bit;
        if(i<j){ complex_t tmp=x[i]; x[i]=x[j]; x[j]=tmp; }
    }

    for(int len=2; len<=n; len<<=1){
        double ang = 2*M_PI/len * (invert ? -1 : 1);
        complex_t wlen = {cos(ang), sin(ang)};
        for(int i=0;i<n;i+=len){
            complex_t w = {1.0,0.0};
            for(int j=0;j<len/2;j++){
                complex_t u = x[i+j];
                complex_t v = c_mul(x[i+j+len/2], w);
                x[i+j] = c_add(u,v);
                x[i+j+len/2] = c_sub(u,v);
                w = c_mul(w, wlen);
            }
        }
    }
    if(invert){
        for(int i=0;i<n;i++){ x[i].re/=n; x[i].im/=n; }
    }
}

/* ===================== FFT 2D ===================== */
void fft2d(complex_t **u, int nx, int ny, int invert){
    #pragma omp parallel for
    for(int i=0;i<nx;i++) fft1d(u[i], ny, invert);

    complex_t *col = malloc(nx*sizeof(complex_t));
    for(int j=0;j<ny;j++){
        for(int i=0;i<nx;i++) col[i]=u[i][j];
        fft1d(col, nx, invert);
        for(int i=0;i<nx;i++) u[i][j]=col[i];
    }
    free(col);
}

/* ===================== MAIN ===================== */
int main(){
    int nthreads = omp_get_max_threads();

    printf(" ECUACION DEL CALOR 2D - FFT PROPIA\n");
    printf("Nx = %d, Ny = %d\n", NX, NY);
    printf("alpha = %g, t_final = %g\n", ALPHA, T_FINAL);
    printf("OpenMP threads (OMP_NUM_THREADS) = %d\n", nthreads);

    double dx = LX/NX, dy = LY/NY;

    complex_t **u = malloc(NX*sizeof(complex_t*));
    for(int i=0;i<NX;i++) u[i]=malloc(NY*sizeof(complex_t));

    /* Condición inicial */
    double sigma = 0.05;

    for(int i=0;i<NX;i++){
        for(int j=0;j<NY;j++){
            double x = i * dx;
            double y = j * dy;
            double r2 = (x - LX/2.0)*(x - LX/2.0)
                    + (y - LY/2.0)*(y - LY/2.0);
            u[i][j].re = exp(-r2/(sigma*sigma));
            u[i][j].im = 0.0;
    }
}


    double t0 = omp_get_wtime();

    /* FFT directa */
    fft2d(u, NX, NY, 0);

    /* Evolución espectral */
    #pragma omp parallel for collapse(2)
    for(int p=0;p<NX;p++){
        for(int q=0;q<NY;q++){
            int kx = (p <= NX/2) ? p : p-NX;
            int ky = (q <= NY/2) ? q : q-NY;
            double k2 = pow(kx/LX,2) + pow(ky/LY,2);
            double factor = exp(-4*M_PI*M_PI*ALPHA*k2*T_FINAL);
            u[p][q].re *= factor;
            u[p][q].im *= factor;
        }
    }

    /* FFT inversa */
    fft2d(u, NX, NY, 1);

    double t1 = omp_get_wtime();

    printf("Tiempo total de ejecucion: %f segundos\n", t1-t0);

    /* Guardar resultados */
    FILE *f = fopen("output_heat.txt","w");
    fprintf(f,"# Solucion ecuacion del calor 2D\n");
    fprintf(f,"# Nx=%d Ny=%d alpha=%g t=%g\n",NX,NY,ALPHA,T_FINAL);
    fprintf(f,"# Columnas: x y u(x,y,t)\n\n");

    for(int i=0;i<NX;i++){
        for(int j=0;j<NY;j++){
            fprintf(f,"%f %f %f\n", i*dx, j*dy, u[i][j].re);
        }
        fprintf(f,"\n");
    }
    fclose(f);

    printf("Resultados guardados en output_heat.txt\n");

    for(int i=0;i<NX;i++) free(u[i]);
    free(u);

    // ------------ GUARDAR EN ARCHIVO ------------------------
    FILE *file = fopen("Tiempos_heat.txt", "a");
    if (file == NULL) {
        printf("Error al abrir el archivo de salida.\n");
        return 1;
    }

    #ifdef _OPENMP
    //fprintf(file, "%d\t%f\t%f\t%f\t%g\t%d\n", N, min_x, min_y, min_f, time, num_threads);
    fprintf(file, "%d\t%f\n", nthreads, t1-t0);
    #endif

    fclose(file);
    return 0;
}
