#include "generate_datasets.h"
#include <iostream>

int mlcppp_make_regression(struct mlccpp_dataset* dataset, unsigned int num_samples, unsigned int num_features, unsigned int num_targets, unsigned int* num_informative, double* bias, double* noise, unsigned int* rand_seed)
{
    if(dataset == nullptr)
    {
        return -1;
    }

    double* features_ptr = reinterpret_cast<double*>(malloc(num_samples * num_features * (sizeof *features_ptr)));
    double* targets_ptr = reinterpret_cast<double*>(malloc(num_samples * num_targets * (sizeof *targets_ptr)));
    double* coefs_ptr = reinterpret_cast<double*>(malloc(num_features * num_targets * (sizeof *coefs_ptr)));
    double* bias_ptr = reinterpret_cast<double*>(malloc(num_targets * (sizeof *bias_ptr)));

    if(features_ptr == nullptr || targets_ptr == nullptr || coefs_ptr == nullptr || bias_ptr == nullptr){
        free(features_ptr);
        free(targets_ptr);
        free(coefs_ptr);
        free(bias_ptr);
        return -1;
    }

    auto [features, targets, coefs, _bias] =  mlccpp::datasets::RegressionDataset::CreateDataset(
        num_samples, 
        num_features,
        num_targets,
        (bias == nullptr) ? std::span<double>() : std::span<double>(bias, num_targets),
        (noise == nullptr)? std::optional<double>() : *noise,
        (num_informative == nullptr) ? std::optional<int>() : *num_informative,
        (rand_seed == nullptr)? std::optional<int>() : *rand_seed
    );

    std::copy(features.data(), features.data() + features.size(), features_ptr);
    std::copy(targets.data(), targets.data() + targets.size(), targets_ptr);
    std::copy(coefs.data(), coefs.data() + coefs.size(), coefs_ptr);
    std::copy(_bias.data(), _bias.data() + _bias.size(), bias_ptr);

    dataset->num_samples = num_samples;
    dataset->num_features = num_features;
    dataset->num_targets = num_targets;
    dataset->features = features_ptr;
    dataset->targets = targets_ptr;
    dataset->coefs = coefs_ptr;
    dataset->bias = bias_ptr;

    return 0;
}