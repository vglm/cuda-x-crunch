import os
import subprocess
import time
import logging

from process import run_process

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

gpus = []

total_accepted_addresses = 0

def check_address(address):
    global total_accepted_addresses
    if len(address) != 42:
        raise ValueError("Address length is not 42")
    if address[0:2] != "0x":
        raise ValueError("Address does not start with 0x")
    total_accepted_addresses += 1

    addr_str = address[2:]

def get_gpu_info():
    kernel_group_size = 128
    kernel_group_count = 5000
    kernel_rounds = 2000
    private_kernel_group_size = 64
    private_kernel_group_count = 2000
    private_kernel_rounds = 1000
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error retrieving GPU info:", result.stderr)
            return

        _gpus = result.stdout.strip().split("\n")
        for i, gpu in enumerate(_gpus):
            name, memory, driver, compute_cap = gpu.split(", ")
            print(f"GPU {i}:")
            print(f"  Name: {name}")
            print(f"  Total Memory: {memory} MB")
            print(f"  Driver Version: {driver}")
            print(f"  Compute Capability: {compute_cap}")
            print("-" * 40)

            if "1080ti" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 128
                kernel_group_count = 1000
                kernel_rounds = 2000
            elif "1080" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 128
                kernel_group_count = 1000
                kernel_rounds = 2000

            if "RTX 2060" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 128
                kernel_group_count = 1000
                kernel_rounds = 2000
            if "RTX 2070" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 128
                kernel_group_count = 1000
                kernel_rounds = 2000
            if "RTX 2080" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 128
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 2090" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 5000
                kernel_rounds = 2000

            if "RTX 3050" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 3060" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 3070" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 3080" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 4000
                kernel_rounds = 2000
            if "RTX 3090" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_count = 5000
                kernel_rounds = 2000

            if "RTX 4050" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 4060" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 4070" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 4080" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 4000
                kernel_rounds = 4000
            if "RTX 4090" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 5000
                kernel_rounds = 5000

            if "RTX 5050" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 5060" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 5070" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 2000
                kernel_rounds = 2000
            if "RTX 5080" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 4000
                kernel_rounds = 4000
            if "RTX 5090" in name:
                print(f"{name} detected. Adjusting kernel parameters.")
                kernel_group_size = 256
                kernel_group_count = 5000
                kernel_rounds = 5000

            create3_find_args = f"-k {kernel_group_size} -g {kernel_group_count} -r {kernel_rounds}"
            private_find_args = f"-k {private_kernel_group_size} -g {private_kernel_group_count} -r {private_kernel_rounds}"


            gpus.append({
                "index": i,
                "name": name,
                "memory": memory,
                "driver": driver,
                "compute_cap": compute_cap,
                "create3_find_args": create3_find_args,
                "private_find_args": private_find_args
            })
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure you have NVIDIA drivers and CUDA installed.")

last_print_time = time.time()
def decode_output(process, stdout_line):
    global gpus
    global total_accepted_addresses
    try:
        idx = process.index
    except Exception as ex:
        print("Index not found in process")
        return
    if "Total compute" in stdout_line:
        total_compute = float(stdout_line.split("Total compute")[1].split("GH")[0].strip())
        reported_speed = float(stdout_line.split("GH -")[1].split("MH")[0].strip())
        gpus[idx]["reported_speed"] = reported_speed
        gpus[idx]["total_compute"] = total_compute

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
        logger.debug("GPU no: {} Accepted: {} Total {}G Speed {:.0f}MH/s".format(idx, total_accepted_addresses, gpus[idx]["total_compute"], gpus[idx]["reported_speed"]))


def decode_error(process, stdout_line):
    decode_output(process, stdout_line)


if __name__ == "__main__":
    get_gpu_info()

    max_time = 60
    factory = os.environ.get('FACTORY', None)
    factory_switch = f"-f {factory}" if factory else ""
    public_key_base = os.environ.get('KEY_BASE', None)
    public_key_switch = f"-z {public_key_base}" if public_key_base else ""
    if public_key_switch != "" and factory_switch != "":
        raise Exception("Cannot specify both KEY_BASE and FACTORY")

    for gpu in gpus:

        command = "profanity_cuda -b {} --device {} {}{}{}".format(max_time, gpu['index'], gpu["create3_find_args"], factory_switch, public_key_switch)
        print(command)
        process, stdout_thread, stderr_thread = run_process(command.split(" "), decode_output, decode_error)
        process.index = gpu['index']

        gpu['process'] = process
        gpu['stdout_thread'] = stdout_thread
        gpu['stderr_thread'] = stderr_thread


    for gpu in gpus:
        gpu['process'].wait()
        gpu['stdout_thread'].join()
        gpu['stderr_thread'].join()
        print(f"GPU {gpu['index']} finished")
