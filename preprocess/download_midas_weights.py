import os
import urllib.request
import torch.hub

import os
import urllib.request
import torch

import os
import urllib.request
import torch

def download_midas_weights(model_name="dpt_large_384"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "midas", "midas", "weights")
    os.makedirs(model_dir, exist_ok=True)

    model_urls = {
        "dpt_large_384": (
            "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt",
            "dpt_large_384.pt"
        ),
    }

    if model_name not in model_urls:
        raise ValueError(f"Unsupported model: {model_name}")

    url, filename = model_urls[model_name]
    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_path):
        print(f"⬇️ Downloading {model_name} weights...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to: {model_path}")
    else:
        print(f"Found existing weights at: {model_path}")

    return model_path


if __name__ == "__main__":
    download_midas_weights()
    