#include <stdlib.h>

#ifndef ML_C_CPP_UTILS_TRAIN_TEST_SPLIT_HEADER
#define ML_C_CPP_UTILS_TRAIN_TEST_SPLIT_HEADER


#ifdef __cplusplus
extern "C" {
#endif

struct train_test_split
{
    double* train_features;
    double* test_features;
    double* train_targets;
    double* test_targets;
};


// int mlccpp_train_test_split(
//     struct train_test_split& split, 
//     double train_size,
//     double test_size,
//     int shuffle,
//     unsigned int* randseed
// );

// C function declaration and headers

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <Eigen/Dense>
namespace mlccpp
{
    namespace utils
    {
        
    } // namespace 
    
    
} // namespace mlccpp

// C++ only headers

#endif //C++
#endif //ML_C_CPP_UTILS_TRAIN_TEST_SPLIT_HEADER