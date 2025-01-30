import os

exe_file = "Release\\profanity_cuda.exe" if os.name == 'nt' else "profanity_cuda"

os.system('"{}" -b'.format(exe_file))