import os

os.system("cmake .")

os.system("git describe --tags > ver.txt")
with open("ver.txt", "r") as f:
    version = f.read().strip()

version = version.replace("v", "")

with open("version.h", "w") as f:
    f.write("#pragma once\n")
    f.write("#include <string>\n")
    f.write('const std::string g_strVersion = "{}";'.format(version))

if os.name == 'nt':
    # msbuild_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
    msbuild_path = r"MSBuild.exe"
    os.system('"{}" profanity_cuda.sln /p:Platform=x64 /p:Configuration=Release /t:profanity_cuda'.format(msbuild_path))
else:
    os.system("make")

exe_file = "Release\\profanity_cuda.exe" if os.name == 'nt' else "profanity_cuda"

os.system('"{}" -b'.format(exe_file))