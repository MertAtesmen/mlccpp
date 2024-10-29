#include "sgd_regressor.h"
#include <iostream>

#ifndef ML_C_CPP_LINEAR_SGD_REGRESSOR_IMPLEMENTATION_HEADER
#define ML_C_CPP_LINEAR_SGD_REGRESSOR_IMPLEMENTATION_HEADER

namespace mlccpp
{
    namespace linear
    {
        SGDRegressor::SGDRegressor(const sgd_regressor_model_params& params)
        : _params{params}
        {
        }

        SGDRegressor::~SGDRegressor()
        {
        }

        inline void SGDRegressor::initialize_weights(size_t num_features, size_t num_targets)
        {
            _coefs = MatrixRowMajorType::Zero(Eigen::Index(num_features), Eigen::Index(num_targets));
            _bias = VectorRowMajorType::Zero(Eigen::Index(num_targets));
        }

        template<typename VectorLike1, typename VectorLike2>
        double SGDRegressor::update_weights(const VectorLike1& features, const VectorLike2& target)
        {
            VectorRowMajorType loss = (features * _coefs  + _bias) - target;

            // Update the weights
            _coefs += (-_params.learning_rate) * 2 * features.transpose() * loss;

            // Update the bias
            _bias += (-_params.learning_rate) * 2  * loss;

            return loss.mean();
        }

        // void SGDRegressor::fit(const double** features, const double *targets, int num_samples, int num_features)
        // {
        //     initialize_weights(num_features);

        //     for (std::size_t iteration = 0; iteration < _params.num_iterations; iteration++)
        //     {
        //         for (std::size_t sample_index = 0; sample_index < num_samples; sample_index++)
        //         {
        //             update_weights(
        //                 Eigen::Map<Eigen::VectorXd>(const_cast<double*>(features[sample_index]), num_features), 
        //                 targets[sample_index]
        //             );
        //         }
        //     }
            
        // }

        // void SGDRegressor::fit(const double* features, const double *targets, int num_samples, int num_features)
        // {
        //     initialize_weights(num_features);

        //     for (std::size_t iteration = 0; iteration < _params.num_iterations; iteration++)
        //     {
        //         for (std::size_t sample_index = 0; sample_index < num_samples; sample_index++)
        //         {
        //             update_weights(
        //                 Eigen::Map<Eigen::VectorXd>(const_cast<double*>(features + (num_features * iteration)), num_features), 
        //                 targets[sample_index]
        //             );
        //         }
        //     }
        // }

        template<typename MatrixLike1, typename MatrixLike2>
        void SGDRegressor::fit(const MatrixLike1& features, const MatrixLike2& targets)
        {   
            initialize_weights(features.cols(), targets.cols());

            for (std::size_t iteration = 0; iteration < _params.num_iterations; iteration++)
            {
                for (std::size_t sample_index = 0; sample_index < features.rows(); sample_index++)
                {
                    update_weights(
                        features.row(sample_index),
                        targets.row(sample_index)
                    );
                }
            }
        }

        template<typename VectorLike>
        VectorRowMajorType SGDRegressor::predict(const VectorLike& features)
        {
            VectorRowMajorType prediction = features * _coefs  + _bias;
            return prediction;
        }

        const MatrixRowMajorType& SGDRegressor::get_coefs() const
        {
            return _coefs;
        }
        
        const VectorRowMajorType& SGDRegressor::get_bias() const
        {
            return _bias;
        }

        inline std::tuple<size_t, size_t> SGDRegressor::get_coef_size()
        {
            return {_coefs.rows(), _coefs.cols()};
        }

        inline size_t SGDRegressor::get_bias_size()
        {
            return _bias.size();
        }

        inline void SGDRegressor::set_weights(const MatrixRowMajorType &coefs, const VectorRowMajorType &bias)
        {
            _coefs = coefs;
            _bias = bias;
        }

        void SGDRegressor::set_params(sgd_regressor_model_params &new_params)
        {
            _params = new_params;
        }
        const sgd_regressor_model_params &SGDRegressor::get_params() const
        {
            return _params;
        }
    }
}

#endif //ML_C_CPP_LINEAR_SGD_REGRESSOR_IMPLEMENTATION_HEADER