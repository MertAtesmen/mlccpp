#include <iostream>
#include <array>
#include <algorithm>
#include <random>
#include <memory>

#include <string>
#include <sstream>
#include <span>
#include <numeric>
#include <functional>

#include <chrono>

#include "mlccpp/linear/sgd_regressor.h"
#include "mlccpp/datasets/generate_datasets.h"

constexpr unsigned int SAMPLE_SIZE = 10000;
constexpr unsigned int FEATURE_COUNT = 10;
constexpr unsigned int TARGET_COUNT = 1;
constexpr bool TIMING = false;


int main(int argc, char const *argv[])
{

    std::vector biasses = {2.0};

    mlccpp::datasets::RegressionDataset dataset(
        SAMPLE_SIZE, FEATURE_COUNT, TARGET_COUNT, biasses, 1.0, {}, 42
    );

    const mlccpp::datasets::MatrixRowMajorType& features = dataset.get_features();
    const mlccpp::datasets::MatrixRowMajorType& target = dataset.get_targets();
    const mlccpp::datasets::VectorRowMajorType& bias = dataset.get_bias();
    const mlccpp::datasets::MatrixRowMajorType& coefs = dataset.get_coefs();

    std::cout << "Coefs:\n" << coefs << '\n';
    std::cout << "Biases:\n" << bias << '\n';


    mlccpp::linear::SGDRegressor regressor{sgd_regressor_model_params{.learning_rate = 0.01, .num_iterations=1000}};

    if (TIMING){
        for(std::size_t i = 0; i < 10; ++i)
        {    
            auto start = std::chrono::high_resolution_clock::now();

            regressor.fit(features, target);

            auto end = std::chrono::high_resolution_clock::now();

            // Calculate elapsed time in milliseconds
            std::chrono::duration<double, std::milli> elapsed = end - start;

            std::cout << "Time taken: " << elapsed.count() << " ms" << '\n';
        }
    }
    else{
        regressor.fit(features, target);
    }

    std::cout << "Coefs:\n" << regressor.get_coefs() << '\n';
    std::cout << "Biases:\n" << regressor.get_bias() << '\n';


    return 0;
}
