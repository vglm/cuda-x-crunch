import os
import time
import signal
from process import run_process

def decode_output(process, stdout_line):
    print("  decoder stdout {} :".format(process.pid), stdout_line)

# Example usage:
command = ["docker", "run", "--gpus", "all", "--name", "cuda_test", "--rm", "-t", "profanity_cuda", "profanity_cuda", "-b", "10"]
process, stdout_thread, stderr_thread = run_process(command, decode_output)
print("Process started with PID:", process.pid)


time.sleep(5)
print("Stop docker container...")
os.system("docker stop cuda_test -t 0")

print("Process {} closed...".format(process.pid))
stdout_thread.join()
stderr_thread.join()



