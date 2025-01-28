
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

add_library(particles STATIC
  ArgParser.hpp
  keccak.cu
  keccak.h
  create3.cu
  create3.h
  precomp.cpp
  precomp.hpp
  types.hpp
  utils.hpp
  utils.cpp
  debug_utils.hpp
  debug_utils.cpp
  )


# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)

  add_executable(profanity_cuda profanity.cu)

  set_target_properties(profanity_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(profanity_cuda PRIVATE particles)

endif()