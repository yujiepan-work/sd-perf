import random
import socket
import time
from dataclasses import asdict, dataclass, field
import datetime
from pathlib import Path
import json
import itertools

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from optimum.intel.openvino import OVStableDiffusionPipeline

PROMPT = "a photo of an astronaut riding a horse on mars"
DEVICE="cpu" # "cpu" or "gpu"
prof_unet_nstep = 20
warmup_loops = 40
prof_nloop = 20


def set_seed(seed:int=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def json_dump(obj, file_path):
    with open(Path(file_path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

@dataclass
class DiffusionPipelineBenchmarkResult:
    latency: float
    latencies: list[float]
    safety_check: bool
    prompt: str
    shape: list[int]
    batch_size: int = 1
    num_images_per_prompt: int = 1
    num_inference_steps: int = 20
    warmup_loops: int = 20
    actual_loops: int = 20
    output_type: str = 'np'
    hostname: str = 'none'
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))

    def to_dict(self):
        return asdict(self)

def get_elapsed_time(pipeline: OVStableDiffusionPipeline, prompt: str, nb_pass=20, num_inference_steps=20, warmup_loops=20):
    #warmup
    print("[Info]: Warmup")
    for _ in range(warmup_loops):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np").images[0]
    print("[Info]: Measuring")
    latencies = []
    for _ in range(nb_pass):
        start = time.perf_counter()
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np").images[0]
        end = time.perf_counter()
        latencies.append(end-start)
        print(f'>>> Latency={latencies[-1]}')
    return latencies

def create_openvino_sd_pipe(model_id, keyname: str, device, is_torch_model: bool, h=512, w=512):
    print("[Info]: Create OV pipeline with model_id: {} on device: {}".format(model_id, device))
    dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    folder = Path('./models', keyname)
    need_save = not folder.exists()
    need_export = is_torch_model and not folder.exists()
    if not need_save:
        pipe = OVStableDiffusionPipeline.from_pretrained(folder.as_posix(), scheduler=dpm, device=device, compile=False)
    else:
        pipe = OVStableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm, device=device, compile=False, export=need_export)
    pipe: OVStableDiffusionPipeline
    pipe.reshape(batch_size=1, height=h, width=w, num_images_per_prompt=1)
    pipe.ov_config.update({"PERFORMANCE_HINT": "LATENCY"})
    print('ovconfig:', pipe.ov_config)
    print('safety check:', pipe.safety_checker)
    pipe.compile()
    if need_save:
        print('Saving to:', folder)
        pipe.save_pretrained(folder.as_posix())
    return pipe



torch_models = {
    'new-compvis--stable-diffusion-v1-4': 'CompVis/stable-diffusion-v1-4',
    'new-svjack--Stable-Diffusion-Pokemon-en': 'svjack/Stable-Diffusion-Pokemon-en'
}

ov_models = {
    'old-openvino--stable-diffusion-pokemons-fp32': 'OpenVINO/stable-diffusion-pokemons-fp32',
}

t2i_sd_latency = dict()
for k, model_id in itertools.chain(torch_models.items(), ov_models.items()):
    set_seed()
    pipe = create_openvino_sd_pipe(model_id, keyname=k, device=DEVICE, h=512, w=512, is_torch_model=k in torch_models)
    print(f'prompt=<{PROMPT}>')
    latencies = get_elapsed_time(pipe, PROMPT, nb_pass=prof_nloop, num_inference_steps=prof_unet_nstep, warmup_loops=warmup_loops)
    t2i_sd_latency[k] = DiffusionPipelineBenchmarkResult(
        latency=sum(latencies) / len(latencies),
        latencies=latencies,
        safety_check=pipe.safety_checker is not None,
        prompt=PROMPT,
        batch_size=1,
        shape=[512, 512],
        num_inference_steps=prof_unet_nstep,
        hostname=socket.gethostname(),
        warmup_loops=warmup_loops,
        actual_loops=prof_nloop,
    ).to_dict()
    json_dump(t2i_sd_latency, './xxx.json')