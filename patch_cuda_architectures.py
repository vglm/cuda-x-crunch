import os
import re

cmake_file = "CMakeLists.txt"
cuda_archs = os.getenv("CUDA_ARCHITECTURES")

if not cuda_archs:
    print("CUDA_ARCHITECTURES environment variable is not set.")
    exit(1)

with open(cmake_file, "r") as file:
    cmake_content = file.readlines()

pattern = re.compile(r"set\(CMAKE_CUDA_ARCHITECTURES .*\)")
new_line = f"set(CMAKE_CUDA_ARCHITECTURES {cuda_archs})\n"

updated_content = []
found = False
for line in cmake_content:
    if pattern.match(line):
        updated_content.append(new_line)
        found = True
    else:
        updated_content.append(line)

if not found:
    raise Exception("set(CMAKE_CUDA_ARCHITECTURES ...) line not found in CMakeLists.txt")

# Write the updated content back to CMakeLists.txt
with open(cmake_file, "w") as file:
    file.writelines(updated_content)

print("CMakeLists.txt has been updated successfully with CMAKE_CUDA_ARCHITECTURES {}".format(cuda_archs))