import os

os.system("cmake .")
if os.name == 'nt':
    msbuild_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
    os.system('"{}" profanity_cuda.sln /p:Platform=x64 /p:Configuration=Release /t:profanity_cuda'.format(msbuild_path))
else:
    os.system("make")

exe_file = "Release\\profanity_cuda.exe" if os.name == 'nt' else "profanity_cuda"

os.system('"{}" -b'.format(exe_file))