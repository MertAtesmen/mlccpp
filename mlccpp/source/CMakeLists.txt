add_library(mlccpp STATIC
    mlccpp/linear/sgd_regressor.cpp
    mlccpp/datasets/generate_datasets.cpp
    mlccpp/datasets/regression_dataset.cpp
)

target_include_directories(mlccpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})