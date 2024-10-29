#include <stdlib.h>

#ifndef ML_C_CPP_UTILS_UTILS_HEADER
#define ML_C_CPP_UTILS_UTILS_HEADER


#ifdef __cplusplus
extern "C" {
#endif

struct size_t_pair{
    size_t first;
    size_t second;
};

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
        using MatrixRowMajorType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using VectorRowMajorType = Eigen::RowVector<double, Eigen::Dynamic>;
        using MatrixRowMajorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        using VectorRowMajorMap = Eigen::Map<Eigen::RowVector<double, Eigen::Dynamic>>;
    } // namespace 
    
    
} // namespace mlccpp

// C++ only headers

#endif //C++
#endif //ML_C_CPP_UTILS_UTILS_HEADER