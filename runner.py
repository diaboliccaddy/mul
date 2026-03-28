import torch
from tqdm import tqdm
import os
from utils_loc import misc
from feature_extractors import mulsen_features
import pandas as pd
from dataset import get_data_loader
from models.models import Model, Mlp
import time
class Tester():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size

        self.methods = {}

        method = args.method_name.strip()   # 🔥 IMPORTANT

        print("SELECTED METHOD:", method)

        if method == 'PC+RGB+Infra+qformer':
            self.methods['PC+RGB+Infra+qformer'] = mulsen_features.TripleRGBInfraPointFeatures(args)

        elif method == 'PC+RGB+qformer':
            self.methods['PC+RGB+qformer'] = mulsen_features.PCRGBGatingFeatures(args)

        elif method == 'PC+Infra+qformer':
            self.methods['PC+Infra+qformer'] = mulsen_features.PCInfraGatingFeatures(args)

        elif method == 'RGB+Infra+qformer':
            self.methods['RGB+Infra+qformer'] = mulsen_features.RGBInfraGatingFeatures(args)

        elif method == 'RGB':
            self.methods['RGB'] = mulsen_features.RGBFeatures(args)

        elif method == 'Infra':
            self.methods['Infra'] = mulsen_features.InfraFeatures(args)

        elif method == 'PC':
            self.methods['PC'] = mulsen_features.PCFeatures(args)

        else:
            raise ValueError(f"❌ Unknown method_name: '{method}'")


    def fit(self, class_name):
        
        train_loader = get_data_loader(
            "train",
            class_name=class_name,
            img_size=self.image_size,
            args=self.args
        )

        print(f'Extracting train features for class {class_name}')

        for batch in train_loader:

            # -------- FORCE CORRECT UNPACK --------
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 1:
                    sample = batch[0]
                else:
                    continue
            else:
                sample = batch

            # -------- HANDLE ALL POSSIBLE FORMATS --------
            if isinstance(sample, (list, tuple)) and len(sample) >= 3:
                rgb = sample[0]
                infra = sample[1]
                pc = sample[2]

            elif isinstance(sample, dict):
                rgb = sample.get("rgb", None)
                infra = sample.get("infra", None)
                pc = sample.get("pc", None)

            else:
                print("❌ BAD SAMPLE FORMAT:", type(sample))
                continue

            # -------- SAFETY CHECK --------
            if rgb is None or infra is None or pc is None:
                print("❌ Missing modality — skipping sample")
                continue

            sample_fixed = (rgb, infra, pc)

            # -------- ADD TO MEMORY --------
            for method in self.methods.values():
                method.add_sample_to_mem_bank(sample_fixed)

        # -------- VERIFY BEFORE CORESET --------
        for method_name, method in self.methods.items():
            print(f'\nRunning coreset for {method_name} on class {class_name}...')
            if hasattr(method, "patch_xyz_lib") and len(method.patch_xyz_lib) > 0:
                print("XYZ LIB SIZE:", len(method.patch_xyz_lib))

            if hasattr(method, "patch_rgb_lib") and len(method.patch_rgb_lib) > 0:
                print("RGB LIB SIZE:", len(method.patch_rgb_lib))

            if hasattr(method, "patch_infra_lib") and len(method.patch_infra_lib) > 0:
                print("INFRA LIB SIZE:", len(method.patch_infra_lib))

            method.run_coreset()


    def evaluate(self, class_name, output_dir):
        metrics_data = []

        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)

        # Paths for predict maps
        rgb_paths = []
        infra_paths = []
        pc_paths = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):

                # -------- UNIVERSAL UNPACK --------
                if len(batch) == 4:
                    sample, label, pixel_mask, paths = batch
                else:
                    sample = batch[0]
                    label = batch[1]
                    pixel_mask = batch[2]
                    paths = batch[3]

                rgb = sample[0]
                infra = sample[1]
                pc = sample[2]

                sample = (rgb, infra, pc)

                rgb_paths.append(paths[0])
                infra_paths.append(paths[1])
                pc_paths.append(paths[2])

                for method in self.methods.values():
                    method.predict(sample, label, pixel_mask)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            metrics = {
                "Method": method_name,
                "Image_ROCAUC": round(method.image_rocauc, 3),
                "RGB_Pixel_ROCAUC": round(method.rgb_pixel_rocauc, 3),
                "RGB_Pixel_F1": round(method.rgb_pixel_f1, 3),
                "RGB_Pixel_AUPR": round(method.rgb_pixel_aupr, 3),
                "RGB_Pixel_AP": round(method.rgb_pixel_ap, 3),
                "Infra_Pixel_ROCAUC": round(method.infra_pixel_rocauc, 3),
                "Infra_Pixel_F1": round(method.infra_pixel_f1, 3),
                "Infra_Pixel_AUPR": round(method.infra_pixel_aupr, 3),
                "Infra_Pixel_AP": round(method.infra_pixel_ap, 3),
                "PC_Pixel_ROCAUC": round(method.pc_pixel_rocauc, 3),
                "PC_Pixel_F1": round(method.pc_pixel_f1, 3),
                "PC_Pixel_AUPR": round(method.pc_pixel_aupr, 3),
            }
 
            metrics_data.append(metrics)

            print(f'Method:{method_name}, Class: {class_name}, Image ROCAUC: {method.image_rocauc:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, RGB Pixel ROCAUC: {method.rgb_pixel_rocauc:.3f}, F1: {method.rgb_pixel_f1:.3f}, AUPR: {method.rgb_pixel_aupr:.3f}, AP: {method.rgb_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, Infra Pixel ROCAUC: {method.infra_pixel_rocauc:.3f}, F1: {method.infra_pixel_f1:.3f}, AUPR: {method.infra_pixel_aupr:.3f}, AP: {method.infra_pixel_ap:.3f}')
            print(f'Method:{method_name}, Class: {class_name}, PC Pixel ROCAUC: {method.pc_pixel_rocauc:.3f}, F1: {method.pc_pixel_f1:.3f}, AUPR: {method.pc_pixel_aupr:.3f}')

        metrics_df = pd.DataFrame(metrics_data)

        # prediction maps
        # if "RGB" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="rgb")
        # if "PC" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="xyz")
        # if "Infra" in self.args.method_name:
        #     method.save_prediction_maps(output_dir, rgb_paths, infra_paths, pc_paths, save_num=10, mode="infra")


        return metrics_df