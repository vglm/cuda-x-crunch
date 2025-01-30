import os

os.system("git describe --tags > ver.txt")
with open("ver.txt", "r") as f:
    version = f.read().strip()

version = version.replace("v", "")

with open("version.h", "w") as f:
    f.write("#pragma once\n")
    f.write('const std::string g_strVersion = "{}";'.format(version))
