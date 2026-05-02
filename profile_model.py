import torch
import time
import numpy as np
from argparse import Namespace

from fe_method.main import (
    TripleRGBInfraPointFeatures
)

# =====================================================
# SETTINGS
# =====================================================

DEVICE = "cuda"


# =====================================================
# ARGS
# =====================================================

args = Namespace(

    method_name='PC+RGB+Infra+qformer',

    memory_bank='multiple',

    rgb_backbone_name='vit_base_patch8_224.dino',

    xyz_backbone_name='Point_MAE',

    fusion_module_path='checkpoints/checkpoint-0.pth',

    save_feature=False,

    save_preds=False,

    group_size=128,

    num_group=1024,

    random_state=None,

    dataset_path='./datasets/MulSen_AD/MulSen_AD',

    img_size=224,

    coreset_eps=0.9,

    f_coreset=0.1,

    asy_memory_bank=None,

    ocsvm_nu=0.5,

    ocsvm_maxiter=1000,

    rm_zero_for_project=False,

    total_epochs=1,

    lr=1e-3,

    weight_decay=1e-2,

    output_dir='./output_dir'
)


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...\n")

model = TripleRGBInfraPointFeatures(args)

device = torch.device(
    DEVICE if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

model.eval()


# =====================================================
# PARAMETER COUNT
# =====================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=====================================")
print("MODEL PARAMETERS")
print("=====================================")

print(f"Total Parameters     : {total_params:,}")

print(f"Trainable Parameters : {trainable_params:,}")


# =====================================================
# REAL SAMPLE FROM DATASET
# =====================================================

from dataset import get_data_loader


print("\nLoading one real sample...\n")

loader = get_data_loader(
    "Capsule",
    "test",
    False,
    args
)

sample = next(iter(loader))

rgb = sample[0][0].unsqueeze(0).to(device)

infra = sample[0][1].unsqueeze(0).to(device)

pc = sample[0][2].unsqueeze(0).to(device)


print("RGB Shape   :", rgb.shape)
print("Infra Shape :", infra.shape)
print("PC Shape    :", pc.shape)


# =====================================================
# WARMUP
# =====================================================

print("\nRunning warmup...\n")

with torch.no_grad():

    for _ in range(20):

        try:

            _ = model(rgb, infra, pc)

        except Exception as e:

            print("Warmup forward failed:")
            print(e)

            break


# =====================================================
# INFERENCE SPEED
# =====================================================

print("\nBenchmarking...\n")

timings = []

with torch.no_grad():

    for _ in range(100):

        torch.cuda.synchronize()

        start = time.time()

        try:

            _ = model(rgb, infra, pc)

        except Exception as e:

            print("Forward failed:")
            print(e)

            break

        torch.cuda.synchronize()

        end = time.time()

        timings.append(end - start)


if len(timings) > 0:

    avg_time = np.mean(timings)

    fps = 1.0 / avg_time

    print("=====================================")
    print("INFERENCE SPEED")
    print("=====================================")

    print(f"Average inference time : {avg_time * 1000:.2f} ms")

    print(f"FPS                    : {fps:.2f}")

else:

    print("\nCould not benchmark model forward pass.\n")