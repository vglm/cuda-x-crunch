
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

add_library(particles STATIC
  ArgParser.hpp
  randomize.cpp
  randomize.h
  particle.cu
  particle.h
  precomp.cpp
  precomp.hpp
  types.hpp
  utils.hpp
  utils.cpp
  debug_utils.hpp
  debug_utils.cpp
  v3.cu
  v3.h
  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(particles PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)

  add_executable(profanity_cuda test.cu)

  set_target_properties(profanity_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(profanity_cuda PRIVATE particles)

endif()