#include "sgd_regressor.h"
#include <iostream>
#include <sstream>

struct sgd_regressor_model{
    mlccpp::linear::SGDRegressor regressor;
};

using MatrixRowMajorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using VectorRowMajorMap = Eigen::Map<Eigen::RowVector<double, Eigen::Dynamic>>;

struct sgd_regressor_model* create_sgd_regressor_model(struct sgd_regressor_model_params params)
{
    // TODO: Make sure the regressor is in place created and not copied.
    sgd_regressor_model* model = new sgd_regressor_model{{params}};
    return model;
}


int fit_sgd_regressor_model(
    sgd_regressor_model *model,
    const double *features,
    const double *targets,
    size_t num_samples,
    size_t num_features,
    size_t num_targets
)
{
    MatrixRowMajorMap feature_map { const_cast<double*>(features), num_samples, num_features };
    MatrixRowMajorMap target_map { const_cast<double*>(targets), num_samples, num_targets };
    model->regressor.fit(feature_map, target_map);
    return 0;
}

int predict_sgd_regressor_model(sgd_regressor_model *model, const double *features, double *target)
{
    VectorRowMajorMap feature_map { const_cast<double*>(features), get_num_features_sgd_regressor_model(model) };
    mlccpp::utils::VectorRowMajorType prediction;

    try
    {
        prediction = model->regressor.predict(feature_map);
    }
    catch(const std::exception& e)
    {
        return -1;
    }

    std::copy(prediction.begin(), prediction.end(), target);
    return 0;
}

struct size_t_pair get_coef_len_sgd_regressor_model(sgd_regressor_model *model)
{
    auto [first, second] = model->regressor.get_coef_size();
    return {first, second};
}

size_t get_bias_len_sgd_regressor_model(sgd_regressor_model *model)
{
    return model->regressor.get_bias_size();
}

size_t get_num_features_sgd_regressor_model(sgd_regressor_model *model)
{
    return model->regressor.get_coefs().rows();
}

size_t get_num_targets_sgd_regressor_model(sgd_regressor_model *model)
{
    return model->regressor.get_coefs().cols();
}

char *get_coefs_string_sgd_regressor_model(sgd_regressor_model *model)
{
    std::stringstream temp_io;
    temp_io << model->regressor.get_coefs();

    std::string buffer = std::move(temp_io.str());

    char* str_ptr = reinterpret_cast<char*>(malloc(buffer.size() * sizeof(char)));

    if(str_ptr == nullptr)
        return nullptr;

    std::copy(buffer.begin(), buffer.end(), str_ptr);
    return str_ptr;
}

char *get_bias_string_sgd_regressor_model(sgd_regressor_model *model)
{
    std::stringstream temp_io;
    temp_io << model->regressor.get_bias();

    std::string buffer = std::move(temp_io.str());

    char* str_ptr = reinterpret_cast<char*>(malloc(buffer.size() * sizeof(char)));
    
    if(str_ptr == nullptr)
        return nullptr;

    std::copy(buffer.begin(), buffer.end(), str_ptr);
    return str_ptr;
}

void get_weights_sgd_regressor_model(sgd_regressor_model *model, double *coefs, double *bias)
{
    const mlccpp::utils::MatrixRowMajorType& _coefs = model->regressor.get_coefs();
    const mlccpp::utils::VectorRowMajorType& _bias = model->regressor.get_bias();

    std::copy(_coefs.data(), _coefs.data() + _coefs.size(), coefs);
    std::copy(_bias.begin(), _bias.end(), bias);
}

void destroy_sgd_regressor_model(sgd_regressor_model *model)
{
    delete model;
}
