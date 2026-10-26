"""Download the official Transformers-native ColPali checkpoint locally."""
from huggingface_hub import snapshot_download
from scripts.domain_common.multimodal_rag import ROOT
from scripts.download_model_files import download

MODEL_ID = "vidore/colpali-v1.3-hf"
MODEL_PATH = ROOT / "indexes/models/colpali-v1.3-hf"


def main():
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH, allow_patterns=["*.json"],
                      ignore_patterns=["1_MultiVectorMask/*"])
    for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
        download(MODEL_ID, name, MODEL_PATH)
    print(f"ColPali ready: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
