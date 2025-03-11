import os
import re

# Define the path to CMakeLists.txt
cmake_file = "CMakeLists.txt"

# Get CUDA_ARCHITECTURES from environment variables
cuda_archs = os.getenv("CUDA_ARCHITECTURES")

if not cuda_archs:
    print("CUDA_ARCHITECTURES environment variable is not set.")
    exit(1)

# Read the CMakeLists.txt content
with open(cmake_file, "r") as file:
    cmake_content = file.readlines()

# Replace the set(CMAKE_CUDA_ARCHITECTURES ...) line
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
    print("No matching line found in CMakeLists.txt, adding a new entry.")
    updated_content.append(new_line)

# Write the updated content back to CMakeLists.txt
with open(cmake_file, "w") as file:
    file.writelines(updated_content)

print("CMakeLists.txt has been updated successfully.")