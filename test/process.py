import subprocess
import threading
import sys

def read_stream(process, stream, is_error, output_decoder):
    for line in iter(stream.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        if output_decoder:
            output_decoder(process, decoded_line)
        else:
            if is_error:
                print("  stderr {} :".format(process.pid), decoded_line, file=sys.stderr)
            else:
                print("  stdout {} :".format(process.pid), decoded_line)
    stream.close()

def run_process(command, output_decoder):
    """Runs a command and allows sending signals to terminate it."""
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    # Create separate threads for stdout and stderr
    stdout_thread = threading.Thread(target=read_stream, args=(process, process.stdout, False, output_decoder))
    stderr_thread = threading.Thread(target=read_stream, args=(process, process.stderr, True, None))

    stdout_thread.start()
    stderr_thread.start()

    return process, stdout_thread, stderr_thread
