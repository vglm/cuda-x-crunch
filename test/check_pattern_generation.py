import os
import time
import signal
from process import run_process

def decode_output(process, stdout_line):
    print("  decoder stdout {} :".format(process.pid), stdout_line)

# Example usage:
command = ["Debug\\profanity_cuda.exe", "-b", "60"]
process, stdout_thread, stderr_thread = run_process(command, decode_output)
print("Process started with PID:", process.pid)
time.sleep(5)
print("Sending CTRL+C to the process...")
os.kill(process.pid, signal.CTRL_C_EVENT)

process.wait()
stdout_thread.join()
stderr_thread.join()


