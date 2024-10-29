#include "../utils/utils.h"

#ifndef ML_C_CPP_LINEAR_SGD_REGRESSOR_HEADER
#define ML_C_CPP_LINEAR_SGD_REGRESSOR_HEADER


#ifdef __cplusplus
extern "C" {
#endif

struct sgd_regressor_model;

struct sgd_regressor_model_params
{
    double learning_rate;
    unsigned int num_iterations;
};


struct sgd_regressor_model* create_sgd_regressor_model(struct sgd_regressor_model_params params);

int fit_sgd_regressor_model(
    struct sgd_regressor_model* model,
    const double* features,
    const double* targets,
    size_t num_samples,
    size_t num_features,
    size_t num_targets
);

int predict_sgd_regressor_model(struct sgd_regressor_model* model, const double* features, double* target);

struct size_t_pair get_coef_len_sgd_regressor_model(struct sgd_regressor_model* model);
size_t get_bias_len_sgd_regressor_model(struct sgd_regressor_model* model);

size_t get_num_features_sgd_regressor_model(struct sgd_regressor_model* model);
size_t get_num_targets_sgd_regressor_model(struct sgd_regressor_model* model);

char* get_coefs_string_sgd_regressor_model(struct sgd_regressor_model* model);
char* get_bias_string_sgd_regressor_model(struct sgd_regressor_model* model);


void get_weights_sgd_regressor_model(struct sgd_regressor_model* model, double* coefs, double* bias);
void destroy_sgd_regressor_model(struct sgd_regressor_model* model);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <Eigen/Dense>
#include <tuple>


namespace mlccpp
{
    namespace linear
    {
        using MatrixRowMajorType = mlccpp::utils::MatrixRowMajorType;
        using VectorRowMajorType = mlccpp::utils::VectorRowMajorType;
        class SGDRegressor
        {
        private:
            MatrixRowMajorType _coefs;
            VectorRowMajorType _bias;
            sgd_regressor_model_params _params;

            // Private helper functions, such as initialization, updating weights, etc.
            void initialize_weights(size_t num_features, size_t num_targets);

            template<typename VectorLike1, typename VectorLike2>
            double update_weights(const VectorLike1& features, const VectorLike2& target);

        public:
            // Constructor and Destructor
            SGDRegressor(const sgd_regressor_model_params& params);
            ~SGDRegressor();
            
            template<typename MatrixLike1, typename MatrixLike2>
            void fit(const MatrixLike1& features, const MatrixLike2& targets);

            template<typename VectorLike>
            VectorRowMajorType predict(const VectorLike& features);
            
            const MatrixRowMajorType& get_coefs() const;
            const VectorRowMajorType& get_bias() const;

            std::tuple<size_t, size_t> get_coef_size();
            size_t get_bias_size();
            
            void set_weights(const MatrixRowMajorType& coefs, const VectorRowMajorType& bias);
            void set_params(sgd_regressor_model_params& new_params);
            const sgd_regressor_model_params& get_params() const;
        };
    }
}

#include "sgd_regressor_impl.h"

#endif //C++
#endif //ML_C_CPP_LINEAR_SGD_REGRESSOR_HEADER