
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(profanity_cuda LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 20)

# Set the CUDA architecture (optional)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Modify according to your GPU

add_executable(profanity_cuda
  profanity.cu
  ArgParser.hpp
  keccak.cu
  keccak.h
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
 )

 if(UNIX AND NOT APPLE)
     target_include_directories(profanity_cuda PRIVATE /usr/local/cuda/targets/x86_64-linux/include)
 endif()


# Additional compile options (optional)
#target_compile_options(profanity_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

#set_property(TARGET profanity_cuda PROPERTY CUDA_ARCHITECTURES 72)

