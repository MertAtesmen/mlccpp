#ifndef ML_C_CPP_DATASTES_REGRESSION_DATASET_HEADER
#define ML_C_CPP_DATASTES_REGRESSION_DATASET_HEADER

#ifdef __cplusplus

#include "../utils/utils.h"
#include <Eigen/Dense>
#include <optional>
#include <span>

namespace mlccpp
{
    namespace datasets
    {
        using MatrixRowMajorType = mlccpp::utils::MatrixRowMajorType;
        using VectorRowMajorType = mlccpp::utils::VectorRowMajorType;
        class RegressionDataset
        {
        public:
            // Returns (MatrixRowMajorType Features,
            //  MatrixRowMajorType Targets,
            //  MatrixRowMajorType Coefs,
            //  VectorRowMajorType Biases)
            static std::tuple<MatrixRowMajorType, MatrixRowMajorType, MatrixRowMajorType, VectorRowMajorType> 
            CreateDataset(
                unsigned int num_samples,
                unsigned int num_features,
                unsigned int num_targets,
                std::span<double> bias,
                std::optional<double> noise = std::optional<double>(),
                std::optional<unsigned int> num_informative = std::optional<unsigned int>(),
                std::optional<unsigned int> rand_seed = std::optional<unsigned int>()
            );
            
        public:
            RegressionDataset(
                unsigned int num_samples,
                unsigned int num_features,
                unsigned int num_targets,
                std::span<double> bias,
                std::optional<double> noise = std::optional<double>(),
                std::optional<unsigned int> num_informative = std::optional<unsigned int>(),
                std::optional<unsigned int> rand_seed = std::optional<unsigned int>()
            );
            ~RegressionDataset();
            
            const MatrixRowMajorType& get_features();
            MatrixRowMajorType&& move_features();
            
            const MatrixRowMajorType& get_targets();
            MatrixRowMajorType&& move_targets();

            const MatrixRowMajorType& get_coefs();
            MatrixRowMajorType&& move_coefs();
            
            const VectorRowMajorType& get_bias();
            VectorRowMajorType&& move_bias();
        
        private:
            MatrixRowMajorType _features;
            MatrixRowMajorType _targets;
            MatrixRowMajorType _coefs;
            VectorRowMajorType _bias;
        };
        

    }
}

#endif // __cplusplus
#endif // ML_C_CPP_DATASTES_REGRESSION_DATASET_HEADER