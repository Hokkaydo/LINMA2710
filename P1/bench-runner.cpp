#include "matrix.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {

    // Run 1000x1000 matrix multiplication 1000 times if no arguments are provided
    if(argc == 1) {
        int it_max = 1e3;
        Matrix a(1000, 1000);
        a.fill(1.5);
        Matrix b(1000, 1000);
        b.fill(2.5);
        for(int t = 0; t < it_max; t++) Matrix _ = a*b;
        return 0;
    }

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << 
        " <start_m>:<stop_m>:<step_m> <start_n>:<stop_n>:<step_n> <start_k>:<stop-k>:<step_k>\n";
        return 1;
    }
    int m_start, m_stop, m_step;
    int n_start, n_stop, n_step;
    int k_start, k_stop, k_step;
    sscanf(argv[1], "%d:%d:%d", &m_start, &m_stop, &m_step);
    sscanf(argv[2], "%d:%d:%d", &n_start, &n_stop, &n_step);
    sscanf(argv[3], "%d:%d:%d", &k_start, &k_stop, &k_step);

    std::cout << "m,n,k,time" << std::endl;
    for (int m = m_start; m <= m_stop; m += m_step) {
        for (int n = n_start; n <= n_stop; n += n_step) {
            for (int k = k_start; k <= k_stop; k += k_step) {
                Matrix A(m, n);
                Matrix B(n, k);

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 1.0);
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        A.set(i, j, dis(gen));
                    }
                }
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < k; ++j) {
                        B.set(i, j, dis(gen));
                    }
                }    
                double time = 0;
                int it_max = 10;
                for(int t = 0; t < it_max; t++) {
                    // use omp_get_wtime() to measure time
                    auto start = std::chrono::high_resolution_clock::now();
                    Matrix C = A * B;
                    auto end = std::chrono::high_resolution_clock::now();
                    
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1e9;
                }
                std::cout << m << "," << n << "," << k << "," 
                << time/it_max
                << std::endl;
            }
        }
    }
}