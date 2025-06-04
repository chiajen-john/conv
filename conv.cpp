
#include <iostream>

// 1D convolution macros.
#define I (10)     // Input size.
#define W (3)      // Weight size.
#define O (I - W)  // Output size.

// 2D convolution macros.
#define OROW (4)                 // Number of output rows.
#define OCOL (4)                 // Number of output columns.
#define OFM (2)                  // Number of output feature maps.
#define S (1)                    // Stride size.
#define K (3)                    // Filter size.
#define IROW (OROW * S + K - S)  // Number of input rows.
#define ICOL (OCOL * S + K - S)  // Number of input columns.
#define IFM (3)                  // Number of input feature maps.

#define ARRAY3d(arr, N, R, C, in, ir, ic) (arr[R * C * in + C * ir + ic])
#define ARRAY4d(arr, M, N, R, C, im, in, ir, ic) \
  (arr[N * R * C * im + R * C * in + C * ir + ic])

typedef int dnn_t;

void initialize_array(dnn_t *buffer, int size, int is_zero) {
  for (int i = 0; i < size; i++) {
    buffer[i] = (is_zero) ? 0 : rand() % 10;
  }
  return;
}

int compare_arrays(dnn_t *a, dnn_t *b, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

void print_array(dnn_t *buffer, int size) {
  for (int i = 0; i < size; i++) {
    printf("%d ", buffer[i]);
  }
  printf("\n");
  return;
}

void conv_1d_input_stationary(dnn_t input[I], dnn_t weight[W],
                              dnn_t output[O]) {
  for (int i = 0; i < I; i++) {
    for (int w = 0; w < W; w++) {
      if (i >= w) {
        output[i - w] += input[i] * weight[w];
      }
    }
  }
  return;
}

void conv_1d_output_stationary(dnn_t input[I], dnn_t weight[W],
                               dnn_t output[O]) {
  for (int o = 0; o < O; o++) {
    for (int w = 0; w < W; w++) {
      output[o] += input[o + w] * weight[w];
    }
  }
  return;
}

void conv_1d_weight_stationary(dnn_t input[I], dnn_t weight[W],
                               dnn_t output[O]) {
  for (int w = 0; w < W; w++) {
    for (int o = 0; o < O; o++) {
      output[o] += input[o + w] * weight[w];
    }
  }
  return;
}

void conv_2d_baseline(dnn_t *input, dnn_t *weight, dnn_t *output) {
  for (int m = 0; m < OFM; m++) {
    for (int n = 0; n < IFM; n++) {
      for (int r = 0; r < OROW; r++) {
        for (int c = 0; c < OCOL; c++) {
          for (int kr = 0; kr < K; kr++) {
            for (int kc = 0; kc < K; kc++) {
              ARRAY3d(output, M, OROW, OCOL, m, r, c) +=
                  ARRAY3d(input, N, (IROW), (ICOL), n, (r * S + kr),
                          (c * S + kc)) *
                  ARRAY4d(weight, OFM, IFM, K, K, m, n, kr, kc);
            }
          }
        }
      }
    }
  }
  return;
}

void conv_2d_no_local_reuse(dnn_t *input, dnn_t *weight, dnn_t *output) {
  // Buffer tile sizes.
  const int TM = 1;
  const int TN = 1;
  const int TO = 1;

  // Global buffer.
  dnn_t input_buf[TN][TO * S + K - S][TO * S + K - S];
  dnn_t weight_buf[TM][TN][K][K];
  dnn_t output_buf[TM][TO][TO];

  for (int m = 0; m < OFM; m += TM) {
    for (int r = 0; r < OROW; r += TO) {
      for (int c = 0; c < OCOL; c += TO) {
        // Initialize output.
        for (int tr = r, trr = 0; tr < r + TO; tr++, trr++) {
          for (int tc = c, tcc = 0; tc < c + TO; tc++, tcc++) {
            for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
              output_buf[tmm][trr][tcc] =
                  ARRAY3d(output, OFM, OROW, OCOL, tm, tr, tc);
            }
          }
        }
        for (int n = 0; n < IFM; n += TN) {
          // Read input.
          for (int tr = r * S, trr = 0; tr < (r + TO) * S + K - S;
               tr++, trr++) {
            for (int tc = c * S, tcc = 0; tc < (c + TO) * S + K - S;
                 tc++, tcc++) {
              for (int tn = n, tnn = 0; tn < n + TN; tn++, tnn++) {
                input_buf[tnn][trr][tcc] =
                    ARRAY3d(input, IFM, IROW, ICOL, tn, tr, tc);
              }
            }
          }
          // Read weight.
          for (int wr = 0; wr < K; wr++) {
            for (int wc = 0; wc < K; wc++) {
              for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
                for (int tn = n, tnn = 0; tn < n + TN; tn++, tnn++) {
                  weight_buf[tmm][tnn][wr][wc] =
                      ARRAY4d(weight, OFM, IFM, K, K, tm, tn, wr, wc);
                }
              }
            }
          }
          // Tiled convolution.
          for (int trr = 0; trr < TO; trr++) {
            for (int tcc = 0; tcc < TO; tcc++) {
              for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K; kc++) {
                  for (int tmm = 0; tmm < TM; tmm++) {    // parallel-for.
                    for (int tnn = 0; tnn < TN; tnn++) {  // parallel-for.
                      output_buf[tmm][trr][tcc] +=
                          input_buf[tnn][S * trr + kr][S * tcc + kc] *
                          weight_buf[tmm][tnn][kr][kc];
                    }
                  }
                }
              }
            }
          }
        }
        // Write output.
        for (int tr = r, trr = 0; tr < r + TO; tr++, trr++) {
          for (int tc = c, tcc = 0; tc < c + TO; tc++, tcc++) {
            for (int tm = m, tmm = 0; tm < m + TM; tm++, tmm++) {
              ARRAY3d(output, OFM, OROW, OCOL, tm, tr, tc) =
                  output_buf[tmm][trr][tcc];
            }
          }
        }
      }
    }
  }
  return;
}

int main() {
  const int input_dims = IFM * IROW * ICOL;
  const int weight_dims = OFM * IFM * K * K;
  const int output_dims = OFM * OROW * OCOL;

  dnn_t input[input_dims];
  dnn_t weight[weight_dims];
  dnn_t output1[output_dims];
  dnn_t output2[output_dims];

  initialize_array(input, input_dims, 0);
  initialize_array(weight, weight_dims, 0);
  initialize_array(output1, output_dims, 1);
  initialize_array(output2, output_dims, 1);

  conv_2d_baseline(input, weight, output1);
  conv_2d_no_local_reuse(input, weight, output2);

  if (compare_arrays(output1, output2, output_dims)) {
    printf("Correct!\n");
  } else {
    printf("Incorrect!\n");
  }

  return 0;
}
