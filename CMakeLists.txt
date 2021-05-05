cmake_minimum_required(VERSION 3.0)

project(training_data_loader)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
set(CMAKE_CXX_FLAGS_RELEASE "/O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_definitions(/DEVAL_LEARN)
add_definitions(/DUSE_AVX2)
add_definitions(/DYANEURAOU_ENGINE_NNUE)

add_library(YaneuraOu STATIC
    lib/YaneuraOu/bitboard.cpp
    lib/YaneuraOu/book/apery_book.cpp
    lib/YaneuraOu/book/book.cpp
    lib/YaneuraOu/book/makebook.cpp
    lib/YaneuraOu/book/makebook2015.cpp
    lib/YaneuraOu/book/makebook2019.cpp
    lib/YaneuraOu/book/makebook2021.cpp
    lib/YaneuraOu/engine/yaneuraou-engine/yaneuraou-search.cpp
    lib/YaneuraOu/eval/evaluate.cpp
    lib/YaneuraOu/eval/evaluate_bona_piece.cpp
    lib/YaneuraOu/eval/evaluate_mir_inv_tools.cpp
    lib/YaneuraOu/eval/nnue/evaluate_nnue.cpp
    lib/YaneuraOu/eval/nnue/evaluate_nnue_learner.cpp
    lib/YaneuraOu/eval/nnue/features/half_kp.cpp
    lib/YaneuraOu/eval/nnue/features/half_relative_kp.cpp
    lib/YaneuraOu/extra/sfen_packer.cpp
    lib/YaneuraOu/learn/learner.cpp
    lib/YaneuraOu/learn/learning_tools.cpp
    lib/YaneuraOu/learn/multi_think.cpp
    lib/YaneuraOu/mate/mate.cpp
    lib/YaneuraOu/mate/mate_solver.cpp
    lib/YaneuraOu/mate/mate1ply_without_effect.cpp
    lib/YaneuraOu/misc.cpp
    lib/YaneuraOu/movegen.cpp
    lib/YaneuraOu/movepick.cpp
    lib/YaneuraOu/position.cpp
    lib/YaneuraOu/testcmd/benchmark.cpp
    lib/YaneuraOu/testcmd/mate_test_cmd.cpp
    lib/YaneuraOu/testcmd/normal_test_cmd.cpp
    lib/YaneuraOu/thread.cpp
    lib/YaneuraOu/timeman.cpp
    lib/YaneuraOu/tt.cpp
    lib/YaneuraOu/types.cpp
    lib/YaneuraOu/usi.cpp
    lib/YaneuraOu/usi_option.cpp
)
add_library(training_data_loader SHARED training_data_loader.cpp)
add_executable(training_data_loader_exe training_data_loader.cpp)

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)
target_link_libraries(training_data_loader YaneuraOu)
target_link_libraries(training_data_loader_exe Threads::Threads)
target_link_libraries(training_data_loader_exe YaneuraOu)
install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
