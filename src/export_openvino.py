import sys
import argparse
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from src import config
from src.model import build_model

def export_openvino(checkpoint_path: Path, output_dir: Path, image_size: int = 224):
    """
    Exports a trained PyTorch model to OpenVINO IR format (.xml / .bin)
    for high-efficiency low-power inference on the Intel Core Ultra NPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"[*] Loading PyTorch checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("config", {}).get("model_name", config.MODEL_NAME)

    model = build_model(model_name=model_name, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 1: Export to ONNX or convert directly with OpenVINO
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    try:
        import openvino as ov
        print("[*] OpenVINO detected. Converting PyTorch model directly to OpenVINO IR...")

        core = ov.Core()
        available_devices = core.available_devices
        print(f"[i] OpenVINO Detected Hardware Devices: {available_devices}")

        # Direct PyTorch -> OpenVINO Model conversion
        ov_model = ov.convert_model(model, example_input=dummy_input)

        xml_path = output_dir / "shot_peening_coverage.xml"
        ov.save_model(ov_model, str(xml_path))
        print(f"[OK] Successfully exported OpenVINO Model to: {xml_path}")

        # Check NPU availability
        target_device = "NPU" if "NPU" in available_devices else "CPU"
        print(f"[*] Compiling OpenVINO model for device: '{target_device}'...")
        compiled_model = core.compile_model(ov_model, target_device)

        # Benchmark latency
        print("[*] Benchmarking latency (100 iterations)...")
        input_np = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = compiled_model([input_np])

        t0 = time.time()
        for _ in range(100):
            res = compiled_model([input_np])
        avg_ms = ((time.time() - t0) / 100.0) * 1000.0

        sample_pred = float(res[0][0]) * 100.0
        print(f"\n{'='*50}")
        print(f" Target Device:           {target_device}")
        print(f" Average Inference Time:  {avg_ms:.2f} ms / image")
        print(f" Throughput:              {1000.0 / avg_ms:.1f} FPS")
        print(f" Sample Prediction:       {sample_pred:.2f}% coverage")
        print(f"{'='*50}\n")

    except ImportError:
        print("[!] OpenVINO is not installed. Exporting standard ONNX model as fallback...")
        onnx_path = output_dir / "shot_peening_coverage.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["image"],
            output_names=["coverage_pct_norm"],
            dynamic_axes={"image": {0: "batch_size"}, "coverage_pct_norm": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[OK] Saved ONNX model to: {onnx_path}")
        print("[i] To export to OpenVINO later, run: pip install openvino")

def main():
    parser = argparse.ArgumentParser(description="Export Trained Model for Intel NPU Inference")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--output", type=str, default=str(config.OPENVINO_DIR))
    args = parser.parse_args()

    export_openvino(Path(args.checkpoint), Path(args.output))

if __name__ == "__main__":
    main()
