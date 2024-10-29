#ifndef ML_C_CPP_DATASTES_GENERATE_DATASETS_HEADER
#define ML_C_CPP_DATASTES_GENERATE_DATASETS_HEADER

#ifdef __cplusplus
extern "C" {
#endif

// C function declaration and headers

struct mlccpp_dataset
{
    double* features;
    double* targets;
    double* coefs;
    double* bias;
    unsigned int num_samples;
    unsigned int num_features;
    unsigned int num_targets;
};


int mlcppp_make_regression(struct mlccpp_dataset* dataset, unsigned int num_samples, unsigned int num_features, unsigned int num_targets, unsigned int* num_informative, double* bias, double* noise, unsigned int* rand_seed);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include "regression_dataset.h"

#endif //C++
#endif //ML_C_CPP_DATASTES_GENERATE_DATASETS_HEADER