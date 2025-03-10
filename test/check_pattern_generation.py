import os
import time
import re

from process import run_process

total_compute = 0
total_accepted_addresses = 0
pattern3_found = [0] * 40
old_pattern3_founds = 0

all_patterns3_4zeroes_found = False

def find_pattern_indices(pattern, string):
    return [match.start() for match in re.finditer(pattern, string)]


def accept_pattern(str):
    global old_pattern3_founds
    global pattern3_found
    accepted = False
    ind = find_pattern_indices("0000...0000", str)
    if len(ind) > 0:
        accepted = True
        pattern3_found[ind[0]] += 1

    if not accepted:
        print("Pattern not accepted: {}".format(str))

    pattern3_founds = 0;
    for i in range(40):
        if pattern3_found[i] > 0:
            pattern3_founds += 1

    if old_pattern3_founds != pattern3_founds:
        print("Pattern 3 found {}/30 times".format(pattern3_founds))
    old_pattern3_founds = pattern3_founds
    if old_pattern3_founds == 30:
        global all_patterns3_4zeroes_found
        all_patterns3_4zeroes_found = True


def check_address(address):
    global total_accepted_addresses
    if len(address) != 42:
        raise ValueError("Address length is not 42")
    if address[0:2] != "0x":
        raise ValueError("Address does not start with 0x")
    total_accepted_addresses += 1

    addr_str = address[2:]

    accept_pattern(addr_str)


def decode_output(_process, stdout_line):
    global total_compute
    global total_accepted_addresses
    if "Total compute" in stdout_line:
        total_compute = float(stdout_line.split("Total compute")[1].split("GH")[0].strip())

    if stdout_line.startswith("0x"):
        splitted = stdout_line.split(",")
        if len(splitted) == 4:
            salt = splitted[0]
            address = splitted[1]
            factory = splitted[2]
            check_address(address)

    print("Accepted: {} Total {}G".format(total_accepted_addresses, total_compute))


def decode_error(process, stdout_line):
    print("  decoder stderr {} :".format(process.pid), stdout_line)


# Example usage:
command = ["docker", "run", "--gpus", "all", "--name", "cuda_test", "--rm", "-t", "profanity_cuda", "profanity_cuda", "-r", "5000", "-b", "1000"]
process, stdout_thread, stderr_thread = run_process(command, decode_output, decode_error)
print("Process started with PID:", process.pid)

ret = 0
try:
    start_timer = time.time()
    while True:

        if all_patterns3_4zeroes_found:
            print("All patterns found, stopping...")
            break
        if time.time() - start_timer > 10:
            print("Timeout reached, stopping...")
            raise TimeoutError("Timeout reached")
except Exception as e:
    print(f"An error occurred: {e}")
    raise e
finally:
    print("Stop docker container...")
    os.system("docker stop cuda_test -t 0")

    print("Process {} closed...".format(process.pid))
    stdout_thread.join()
    stderr_thread.join()


