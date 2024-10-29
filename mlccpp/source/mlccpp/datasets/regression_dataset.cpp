#include "regression_dataset.h"
#include <random>
#include <iostream>

namespace mlccpp{
    namespace datasets{

        std::tuple<MatrixRowMajorType, MatrixRowMajorType, MatrixRowMajorType, VectorRowMajorType> 
        RegressionDataset::CreateDataset(
            unsigned int num_samples,
            unsigned int num_features,
            unsigned int num_targets,
            std::span<double> bias,
            std::optional<double> noise,
            std::optional<unsigned int> num_informative,
            std::optional<unsigned int> rand_seed
        )
        {
            std::mt19937 rng(rand_seed.has_value()? rand_seed.value() : std::random_device{}());
            std::uniform_real_distribution<double> feature_dist{ -1.0, 1.0 };
            std::uniform_real_distribution<double> coef_dist {-100.0, 100.0};

            unsigned int _num_informative = num_informative.has_value()? num_informative.value() : num_features / 5;
            
            // Feature Generation with random variables
            MatrixRowMajorType features = MatrixRowMajorType::NullaryExpr(
                Eigen::Index(num_samples), Eigen::Index(num_features),
                [&](Eigen::Index i, Eigen::Index j)
                {
                    return feature_dist(rng);    
                } 
            );

            // TODO coef generation
            // Coef generation will also include feature selection as some of them will be set to zero
            MatrixRowMajorType coefs = MatrixRowMajorType::NullaryExpr(
                Eigen::Index(num_features), Eigen::Index(num_targets),
                [&](Eigen::Index i, Eigen::Index j)
                {
                    return coef_dist(rng);
                } 
            );


            // Bias generation
            MatrixRowMajorType _bias;
            
            if (bias.size() == 0)
            {
                _bias = MatrixRowMajorType::Zero(Eigen::Index(num_samples), Eigen::Index(num_targets));
            }
            else
            {
                _bias = MatrixRowMajorType::NullaryExpr(
                    Eigen::Index(num_samples), Eigen::Index(num_targets), 
                    [&](Eigen::Index i, Eigen::Index j)
                    {
                        return bias[j];
                    } 
                );
            }

            MatrixRowMajorType targets = features * coefs + _bias;
            // If there is noise introduce noice to the dataset;
            if (noise.has_value())
            {
                std::normal_distribution<double> noise_dist{0.0, noise.value()};

                targets = targets.unaryExpr([&](double val){
                    return val + noise_dist(rng);
                });
            }

            return {features, targets, coefs, _bias.row(0)};
        }

        RegressionDataset::RegressionDataset(
            unsigned int num_samples,
            unsigned int num_features,
            unsigned int num_targets,
            std::span<double> bias,
            std::optional<double> noise,
            std::optional<unsigned int> num_informative,
            std::optional<unsigned int> rand_seed
        )
        {
            // Tuple unpacking
            std::tie(_features, _targets, _coefs, _bias) = CreateDataset(
                num_samples, num_features, num_targets, bias, noise, num_informative, rand_seed
            ); 
        }

        RegressionDataset::~RegressionDataset()
        {
        }

        const MatrixRowMajorType &RegressionDataset::get_features()
        {
            return _features;
        }

        MatrixRowMajorType &&RegressionDataset::move_features()
        {
            return std::move(_features);
        }

        const MatrixRowMajorType &RegressionDataset::get_targets()
        {
            return _targets;
        }

        MatrixRowMajorType &&RegressionDataset::move_targets()
        {
            return std::move(_targets);
        }
        
        const MatrixRowMajorType &RegressionDataset::get_coefs()
        {

            return _coefs;
        }

        MatrixRowMajorType &&RegressionDataset::move_coefs()
        {
            return std::move(_coefs);
        }

        const VectorRowMajorType &RegressionDataset::get_bias()
        {
            return _bias;
        }
        
        VectorRowMajorType &&RegressionDataset::move_bias()
        {
            return std::move(_bias);
        }
    }
}
