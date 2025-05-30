
#include <iostream>

#define I (4)  // Input size.
#define W (3)   // Weight size.
#define S (1)   // Strize size.
#define O (I - W + 1) / S   // Output size.
#define N (1)   // Input channels.
#define M (1)   // Output channels.

#define ARRAY3d(arr, N, R, C, in, ir, ic) (arr[R * C * in + C * ir + ic])
#define ARRAY4d(arr, M, N, R, C, im, in, ir, ic) (arr[N * R * C * im + R * C * in + C * ir + ic])


typedef int dnn_t;

void initialize_array(dnn_t *buffer, int size, int is_zero) {
    for (int i = 0; i < size; i++) {
        buffer[i] = (is_zero) ? 0 : rand() % 5;
    }
    return;
}

void print_array(dnn_t *buffer, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
    return;
}

void conv_1d_input_stationary(dnn_t input[I], dnn_t weight[W], dnn_t output[O]) {
    for (int i = 0; i < I; i++) {
        for (int w = 0; w < W; w++) {
            if (i >= w) {
                output[i - w] += input[i] * weight[w];
            }
        }
    }
    return;
}

void conv_1d_output_stationary(dnn_t input[I], dnn_t weight[W], dnn_t output[O]) {
    for (int o = 0; o < O; o++) {
        for (int w = 0; w < W; w++) {
            output[o] += input[o + w] * weight[w];
        }
    }
    return;
}

void conv_1d_weight_stationary(dnn_t input[I], dnn_t weight[W], dnn_t output[O]) {
    for (int w = 0; w < W; w++) {
        for (int o = 0; o < O; o++) {
            output[o] += input[o + w] * weight[w];
        }
    }
    return;
}

void conv_2d_baseline(dnn_t *input, dnn_t *weight, dnn_t *output) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int r = 0; r < O; r++) {
                for (int c = 0; c < O; c++) {
                    for (int kr = 0; kr < W; kr++) {
                        for (int kc = 0; kc < W; kc++) {
                            ARRAY3d(output,M,O,O,m,r,c) += ARRAY3d(input,N,I,I,n,r * S + kr,c * S + kc) * ARRAY4d(weight,M,N,W,W,m,n,kr,kc);
                        }
                    }
                }
            }
        }
    }
    return;
}

// void conv_2d_no_local_reuse(dnn_t *input, dnn_t *weight, dnn_t *output) {
    
//     // Buffer tile sizes.
//     const int TM = 1;
//     const int TN = 1;
//     const int TO = 1;
    
//     // Global buffer.
//     dnn_t input_buf[TN][TO * S + W - S][TO * S + W - S];
//     dnn_t weight_buf[TM][TN][W][W];
//     dnn_t output_buf[TM][TO][TO];

//     for (int m = 0; m < M; m += TM) {
//         for (int n = 0; n < N; n += TN) {
//             for (int r = 0; r < O; r += TO) {
//                 for (int c = 0; c < O; c += TO) {

//                     // Read input.
//                     for (int tn = n, tnn = 0; tn < n + TN; tn++, tnn++) {
//                         for (int tr = r * S + W - S, trr = 0; tr < (r + TO) * S + W - S; tr++, trr++) {
//                             for (int tc = r * S + W - S, tcc = 0; tc < (c + TO) * S + W - S; tc++, tcc++) {
//                                 input_buf[tnn][trr][tcc] = input[N * tn + O * tr + tc];
//                             }
//                         }
//                     }

//                     // Read weight.
//                     for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
//                         for (int tn = n, tnn = 0; tn < n + TN; tn++, tnn++) {
//                             for (int wr = 0; wr < W; wr++) {
//                                 for (int wc = 0; wc < W; wc++) {
//                                     weight_buf[tmm][tnn][wr][wc] = weight[tm][tn][wr][wc];
//                                 }
//                             }
//                         }
//                     }

//                     // Read output.
//                     for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
//                         for (int tr = r, trr = 0; tr < r + TO; tr++, trr++) {
//                             for (int tc = c, tcc = 0; tc < c + TO; tc++, tcc++) {
//                                 output_buf[tmm][trr][tcc] = output[tm][tr][tc];
//                             }
//                         }
//                     }

//                     // Tiled convolution.
//                     for (int tmm = 0; tmm < TM; tmm++) {
//                         for (int tnn = 0; tnn < TN; tnn++) {
//                             for (int trr = 0; trr < TO; trr++) {
//                                 for (int tcc = 0; tcc < TO; tcc++) {
//                                     for (int kr = 0; kr < W; kr++) {
//                                         for (int kc = 0; kc < W; kc++) {
//                                             output_buf[tmm][trr][tcc] += input_buf[tnn][S * trr + kr][S * tcc + kc] * weight_buf[tmm][tnn][kr][kc];
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     }
                    
//                     // Write output.
//                     for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
//                         for (int tr = r, trr = 0; tr < r + TO; tr++, trr++) {
//                             for (int tc = c, tcc = 0; tc < c + TO; tc++, tcc++) {
//                                 output[tm][tr][tc] = output_buf[tmm][trr][tcc];
//                             }
//                         }
//                     }

//                 }
//             }
//         }
//     }


//     return;
// }

int main() {

    const int input_dims = N * I * I;
    const int weight_dims = M * N * W * W;
    const int output_dims = M * O * O;

    dnn_t input[input_dims];
    dnn_t weight[weight_dims];
    dnn_t output[output_dims];

    initialize_array(input, input_dims, 0);
    initialize_array(weight, weight_dims, 0);
    initialize_array(output, output_dims, 1);

    conv_2d_baseline(input, weight, output);

    print_array(input, input_dims);
    print_array(weight, weight_dims);
    print_array(output, output_dims);

    return 0;
}
