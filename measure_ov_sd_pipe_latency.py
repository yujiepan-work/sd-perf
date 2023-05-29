
import torch
import random
import os
import time
import numpy as np
import pandas as pd
import socket
from cpuinfo import get_cpu_info
from diffusers import DPMSolverMultistepScheduler
from optimum.intel.openvino import OVStableDiffusionPipeline

def create_openvino_sd_pipe(model_id, h=512, w=512):
    dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = OVStableDiffusionPipeline.from_pretrained(model_id, scheduler=dpm, compile=False)
    pipe.reshape(batch_size=1, height=h, width=w, num_images_per_prompt=1)
    pipe.compile()
    return pipe

def get_elapsed_time(pipeline, prompt, nb_pass=10, num_inference_steps=20):
    #warmup
    print("[Info]: Warmup")
    for _ in range(2):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
    print("[Info]: Measuring")
    start = time.time()
    for _ in range(nb_pass):
        _ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
    end = time.time()
    return (end - start) / nb_pass

pokemon_model_dict = {
    "fp32-sd-pokemon":                "OpenVINO/stable-diffusion-pokemons-fp32",
    "8bit-sd-pokemon-en":             "OpenVINO/Stable-Diffusion-Pokemon-en-quantized",
    "8bit-sd1.5-pokemon":             "OpenVINO/stable-diffusion-pokemons-1-5-quantized",
    "tome-8bit-sd-pokemon":           "OpenVINO/stable-diffusion-pokemons-tome-quantized",
    "aggresive-8bit-sd-pokemon":      "OpenVINO/stable-diffusion-pokemons-quantized-aggressive",
    "tome-aggresive-8bit-sd-pokemon": "OpenVINO/stable-diffusion-pokemons-tome-quantized-aggressive",
}

t2i_model_dict = {
    "8bit-sd2.1": "OpenVINO/stable-diffusion-2-1-quantized",
}

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# dryrun - primary purpose is to download models and flush the pipeline
DRYRUN_BOOL=False
if DRYRUN_BOOL:
    outdir = "/tmp/dryrun/sd-pokemon"
    os.makedirs(outdir, exist_ok=True)
    for k, model_id in pokemon_model_dict.items():
        print(f"[Info]: Start of {k}")
        pipe = create_openvino_sd_pipe(model_id)
        prompt = "cartoon bird"
        output = pipe(prompt, num_inference_steps=20, output_type="pil")
        output.images[0].save(f"{outdir}/{k}.png")
        print(f"[Info]: End of {k}")

if DRYRUN_BOOL:
    outdir = "/tmp/dryrun/sd-t2i"
    os.makedirs(outdir, exist_ok=True)
    for k, model_id in t2i_model_dict.items():
        print(f"[Info]: Start of {k}")
        pipe = create_openvino_sd_pipe(model_id)
        prompt = "sailing ship in storm by Rembrandt"
        output = pipe(prompt, num_inference_steps=20, output_type="pil")
        output.images[0].save(f"{outdir}/{k}.png")
        print(f"[Info]: End of {k}")
# ----------------------------------------------------------------------------

# Measure

prof_unet_nstep = 20
prof_nloop = 10

pokemon_sd_latency = dict()
for k, model_id in pokemon_model_dict.items():
    print(f"[Info]: target: {k}")
    pipe = create_openvino_sd_pipe(model_id)
    prompt = "cartoon bird"
    t = get_elapsed_time(pipe, prompt, nb_pass=prof_nloop, num_inference_steps=prof_unet_nstep)
    pokemon_sd_latency[k] = t
    print(f"[Info]: End of measurement ---\n\n")

t2i_sd_latency = dict()
for k, model_id in t2i_model_dict.items():
    print(f"[Info]: target: {k}")
    pipe = create_openvino_sd_pipe(model_id)
    prompt = "sailing ship in storm by Rembrandt"
    t = get_elapsed_time(pipe, prompt, nb_pass=prof_nloop, num_inference_steps=prof_unet_nstep)
    t2i_sd_latency[k] = t
    print(f"[Info]: End of measurement ---\n\n")

csv_path = "ovsd.latency.{}-{}.csv".format(socket.gethostname(), get_cpu_info()['brand_raw'].replace(" ", "_"))
pokemon_sd_latency.update(t2i_sd_latency)
df = pd.Series(pokemon_sd_latency, name='latency (sec)').to_frame()
df.to_csv(csv_path)

print(f"[Info]: end of script, see {csv_path}")
