import os
import time
import re
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
from process import run_process

total_compute = 0
total_accepted_addresses = 0
pattern3_found = [0] * 40
total_patterns3_found = 0
reported_speed = 0
heavy_zero_found = 0
start_time = time.time()
def find_pattern_indices(pattern, string):
    return [match.start() for match in re.finditer(pattern, string)]


def accept_pattern(str):
    global total_patterns3_found
    global pattern3_found
    accepted = False

    zero_count = str.count("0")
    if zero_count >= 17:
        global heavy_zero_found
        heavy_zero_found += 1
        accepted = True
        logger.debug("Heavy zero found: {} {}".format(heavy_zero_found, str))

    ind = find_pattern_indices("0000...0000", str)
    if len(ind) > 0:
        accepted = True
        pattern3_found[ind[0]] += 1
        # logger.debug("Pattern 3 found: {}".format(str))

    # if not accepted:
    #   print("Pattern not accepted: {}".format(str))

    pattern3_founds = 0;
    for i in range(40):
        if pattern3_found[i] > 0:
            pattern3_founds += 1

    if total_patterns3_found != pattern3_founds:
        logger.debug("Pattern 3 found {}/30 times".format(pattern3_founds))
    total_patterns3_found = pattern3_founds


def check_address(address):
    global total_accepted_addresses
    if len(address) != 42:
        raise ValueError("Address length is not 42")
    if address[0:2] != "0x":
        raise ValueError("Address does not start with 0x")
    total_accepted_addresses += 1

    addr_str = address[2:]

    accept_pattern(addr_str)

last_print_time = time.time()
def decode_output(_process, stdout_line):
    global total_compute
    global total_accepted_addresses
    global reported_speed
    if "Total compute" in stdout_line:
        total_compute = float(stdout_line.split("Total compute")[1].split("GH")[0].strip())
        reported_speed = float(stdout_line.split("GH -")[1].split("MH")[0].strip())

    if stdout_line.startswith("0x"):
        stdout_split = stdout_line.split(",")
        if len(stdout_split) == 4:
            salt = stdout_split[0]
            address = stdout_split[1]
            factory = stdout_split[2]
            check_address(address)

    global last_print_time
    if time.time() - last_print_time > 5:
        last_print_time = time.time()
        logger.debug("Accepted: {} Total {}G Speed {:.0f}MH/s".format(total_accepted_addresses, total_compute, reported_speed))


def decode_error(process, stdout_line):
    logger.error("  decoder stderr {} : {}".format(process.pid, stdout_line) )

os.system("docker build -t profanity_cuda .")
# Example usage:
command = ["docker", "run", "--gpus", "all", "--name", "cuda_test", "--rm", "-t", "profanity_cuda", "profanity_cuda", "-r", "5000", "-b", "1000"]
process, stdout_thread, stderr_thread = run_process(command, decode_output, decode_error)
logger.info("Process started with PID: {}".format(process.pid))

ret = 0
try:
    start_timer = time.time()
    while True:
        if total_patterns3_found < 10 and total_compute > 3:
            raise ValueError("Not enough addresses found to continue. Need at least 10 addresses for 3GH of compute")

        if total_patterns3_found == 30 and heavy_zero_found > 0:
            logger.info("All patterns found, stopping...")
            break
        if time.time() - start_timer > 180:
            logger.info("Timeout reached, stopping...")
            raise TimeoutError("Timeout reached")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise e
finally:
    print("Stop docker container...")
    os.system("docker stop cuda_test -t 0")

    print("Process {} closed...".format(process.pid))
    stdout_thread.join()
    stderr_thread.join()


