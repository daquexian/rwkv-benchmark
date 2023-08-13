#!/usr/bin/env python3

from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import signal
import concurrent.futures
import random
import threading
import subprocess
import wandb
import argparse
import json
import time
import traceback
import os

parser = argparse.ArgumentParser(description='Run benchmark')
parser.add_argument('--backend', type=str, help='RWKV backend', default='ChatRWKV')
parser.add_argument('--branch', type=str, help='branch of ChatRWKV', default='main')
parser.add_argument('--model', type=str, help='Model path', required=True)
parser.add_argument('--verbose', action='store_true', help='Print command output')
parser.add_argument('-n', type=int, help='Number of runs', required=True)
parser.add_argument('--log-dir', type=str, help='log dir')
parser.add_argument('--timeout', type=int, help='timeout for the instance preparation (pulling image, etc.), in seconds', default=900)
args = parser.parse_args()

if args.log_dir is not None:
    assert not os.path.exists(args.log_dir) or len(os.listdir(args.log_dir)) == 0, f"Log dir {args.log_dir} is not empty"
os.makedirs(args.log_dir, exist_ok=True)

wandb.init()

models = [args.model]

strategies = ['bf16', 'fp16', 'fp32', 'fp16i8', 'fp16 *1', 'fp16 *4', 'fp16 *8', 'fp16 *10']

columns = ['Device'] + strategies

vast_gpu_names = {'1080': 'GTX_1080', '2080Ti': 'RTX_2080_Ti', '3080': 'RTX_3080', '4090': 'RTX_4090'}

devices = ['4090', '3080', '2080Ti', '1080', 'cpu']

table = wandb.Table(columns=columns)

tl = threading.local()
lock = threading.Lock()


class TimeoutError(Exception):
    pass


class NoInstanceError(RuntimeError):
    pass


class Backend(ABC):
    @abstractmethod
    def github_url(self) -> str:
        pass

    @abstractmethod
    def docker_image(self) -> str:
        pass

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def run(self) -> Tuple[float, float]:
        pass

    def basename(self) -> str:
        return os.path.basename(self.github_url())


class ChatRWKV(Backend):
    def github_url(self) -> str:
        return 'https://github.com/BlinkDL/ChatRWKV'

    def docker_image(self) -> str:
        return 'daquexian/cuda-pytorch:cu118-dev-2.0.2'

    def prepare(self) -> None:
        scp('benchmark-custom.py', f'{backend.basename()}/benchmark-custom.py')

    def run(self, model, strategy, mode) -> Tuple[float, float]:
        command = ['python3', f'{backend.basename()}/benchmark-custom.py', '--model', f'{model}', '--strategy', f'"{tl.device_type} {strategy}"', '--custom-cuda-op', '--jit', f'--only-{mode}']
        output = remote_check_output(command)
        latency = float(output.splitlines()[-2].split(' ')[2][:-2])
        mem = float(output.splitlines()[-1].split(' ')[-2])
        return latency, mem


backend = eval(args.backend)()


def prepare_vastai_env(device: str):
    if device == 'cpu':
        output = host_check_output(["vastai", "search", "offers", "cpu_cores_effective>=8", '-o', 'dph', "--raw"])
    else:
        vast_gpu_name = vast_gpu_names[device]
        output = host_check_output(["vastai", "search", "offers", f"gpu_name={vast_gpu_name} cpu_cores_effective>=8 cuda_vers>=11.8", '-o', 'dph', "--raw"])
    output = json.loads(output)
    if len(output) == 0:
        raise NoInstanceError(f"No Vast.ai offers found for {device}")
    best = output[0]["id"]
    log(f"Found best offer {best}")
    output = host_check_output(f"vastai create instance {best} --image {backend.docker_image()} --disk 32 --raw --ssh --direct".split())
    output = json.loads(output)
    instance_id = output["new_contract"]
    log(f"Created instance {instance_id}, checking status..")
    flag = False
    waiting_time = 0
    while not flag:
        time.sleep(10)
        waiting_time += 10
        if waiting_time > args.timeout:
            raise TimeoutError("Timeout waiting for instance to be ready")
        log("Checking status..")
        # too verbose
        output = host_check_output(f"vastai show instances --raw".split())
        output = json.loads(output)
        for instance in output:
            if instance["id"] == instance_id:
                log(f"Instance {instance_id} is {instance['actual_status']}")
                if instance["actual_status"] == "running":
                    tl.ssh_user_and_ip = f'root@{instance["public_ipaddr"]}'
                    tl.ssh_port = instance["ports"]["22/tcp"][0]["HostPort"]
                    tl.instance_id = instance_id
                    flag = True
                    # sleep for a while to make sure the instance is ready
                    time.sleep(5)
                    break

    tl.ssh_prefix = f'ssh -o StrictHostKeyChecking=no -p {tl.ssh_port} {tl.ssh_user_and_ip}'.split()
    remote_check_output(['git', 'clone', backend.github_url()])
    basename = backend.basename()
    if args.branch != 'main':
        if '/' in args.branch:
            user, branch = args.branch.split('/')
            remote_check_output([f'cd {basename} && git remote add daquexian https://github.com/{user}/{basename} && git fetch {user}'])
        remote_check_output([f'cd {basename} && git checkout {args.branch}'])

    backend.prepare()


def scp(src, dst):
    log(f"scp from {src} to {dst} of {tl.ssh_user_and_ip}:{tl.ssh_port}")
    subprocess.check_call(['scp', '-o', 'StrictHostKeyChecking=no', '-P', str(tl.ssh_port), src, f'{tl.ssh_user_and_ip}:{dst}'], stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)


def remote_check_output(command: List[str]):
    command = tl.ssh_prefix + [' '.join(command)]
    return host_check_output(command)


def host_check_output(command: List[str]):
    log(f'Running {" ".join(command)}')
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = ""
    for line in proc.stdout: # type: ignore
        if args.verbose:
            log(line.decode('utf-8').strip())
        stdout += line.decode('utf-8')
    assert proc.wait() == 0, f"Command {' '.join(command)} failed with stdout {stdout}"
    return stdout.strip()


def log(msg):
    if args.log_dir is None:
        return
    with open(os.path.join(args.log_dir, f'{tl.device}.txt'), 'a+') as f:
        f.write(msg + '\n')


def run_on_device(device: str):
    tl.device = device
    try:
        # sleep a different period on every thread to bypass vast.ai rate limit
        sleep_time = random.randint(0, 20)
        log(f"Sleeping for {sleep_time} seconds to bypass vast.ai rate limit")
        time.sleep(sleep_time)
        # refactor it as a contextmanager
        prepare_vastai_env(device)
    except NoInstanceError:
        log(f"No instance found for {device}, skipping")
        return
    except TimeoutError:
        log(f"Timeout when preparing {device}, skipping")
        host_check_output(['vastai', 'destroy', 'instance', str(tl.instance_id)])
        return
    except Exception as e:
        log('Fatal error')
        log(traceback.format_exc())
        try:
            host_check_output(['vastai', 'destroy', 'instance', str(tl.instance_id)])
        except:
            pass
        return
    project_dir = backend.basename()
    tl.device_type = 'cpu' if device == 'cpu' else 'cuda'
    for model in models:
        scp(model, f'{project_dir}/{model}')
        data = [device]
        for strategy in strategies:
            for mode in ['slow']:
                try:
                    latency = 99999999999
                    for _ in range(args.n):
                        this_latency, mem = backend.run(f'{project_dir}/{model}', strategy, mode)
                        log(f'Device: {device}, model: {model}, strategy: {strategy}, mode: {mode}, this_latency: {this_latency}, min_latency: {latency}, mem: {mem}')
                        latency = min(latency, this_latency)
                    data.append(f'{latency * 1000:.0f}ms/{mem:.0f}MB') # type: ignore[reportUnboundVariable]
                except:
                    data.append('N/A')
                    log(f'Failed to run {model} on {device} with {strategy}')
                    log(traceback.format_exc())
        with lock:
            table.add_data(*data)
    host_check_output(['vastai', 'destroy', 'instance', str(tl.instance_id)])


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_on_device, devices)


wandb.log({'Latency and Memory': table})

if args.log_dir is not None:
    artifact = wandb.Artifact(name='log', type='log')
    artifact.add_dir(args.log_dir)
    wandb.log_artifact(artifact)

wandb.finish()
