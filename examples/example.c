#include "mlccpp/linear/sgd_regressor.h"
#include "mlccpp/datasets/generate_datasets.h"
#include "mlccpp/datasets/regression_dataset.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    struct sgd_regressor_model_params params = {.learning_rate = 0.01, .num_iterations = 1000};

    struct mlccpp_dataset regressor_dataset;

    unsigned int rand_seed = 42;
    double noise = 1.0;
    double bias_values[1] = {1000.0};

    mlcppp_make_regression(&regressor_dataset, 10000, 10, 1, NULL, &bias_values, &noise, &rand_seed);


    struct sgd_regressor_model* regressor = create_sgd_regressor_model(params);
    fit_sgd_regressor_model(regressor, regressor_dataset.features, regressor_dataset.targets, regressor_dataset.num_samples, regressor_dataset.num_features, regressor_dataset.num_targets);
    
    struct size_t_pair coef_size = get_coef_len_sgd_regressor_model(regressor);

    char* coefs_str = get_coefs_string_sgd_regressor_model(regressor);
    char* bias_str = get_bias_string_sgd_regressor_model(regressor);

    printf("Coefs:\n %s\n", coefs_str);
    printf("bias_str:\n %s\n", bias_str);


    double coefs[coef_size.first * coef_size.second];
    double bias;

    get_weights_sgd_regressor_model(regressor, coefs, &bias);

    double features[10] = {0.5, 4.0, 2.4, 3.5, -5.2, -0.74, -3.45, -7.3, 9.0, -4.2};

    double result = 0;
    for (size_t i = 0; i < 10; i++)
    {
        result += features[i] * coefs[i];
    }
    result += bias;

    printf("Predicted value for loop: %f\n", result);

    predict_sgd_regressor_model(regressor, features, &result);

    printf("Predicted value sgd prediction: %f\n", result);

    return 0;
}
