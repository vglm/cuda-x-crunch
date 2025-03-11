
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(profanity_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

if(DEFINED ENV{CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})
endif()

add_executable(profanity_cuda
  profanity.cu
  ArgParser.hpp
  private_key.cu
  private_key.h
  private_key_search.cpp
  create3.cu
  create3search.cpp
  create3.h
  precomp.cpp
  precomp.hpp
  types.hpp
  utils.hpp
  utils.cpp
  debug_utils.hpp
  debug_utils.cpp
  scorer.cuh
 )

 if(UNIX AND NOT APPLE)
     target_include_directories(profanity_cuda PRIVATE /usr/local/cuda/targets/x86_64-linux/include)
 endif()


# Additional compile options (optional)
#target_compile_options(profanity_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

#set_property(TARGET profanity_cuda PROPERTY CUDA_ARCHITECTURES 72)

