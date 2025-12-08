import os
import sys
import matplotlib.pyplot as plt
from .utils_extract import extract_settling_curves
from .config import JAR_COORDS

def plot_curves(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[INFO] Extracting curves for {video_name}")
    curves = extract_settling_curves(video_path)

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 6))

    for j, curve in enumerate(curves):
        if len(curve) == 0:
            continue
        plt.plot(curve, label=f"Jar {j}")

    plt.title(f"Settling Curves — {video_name}")
    plt.xlabel("Frame")
    plt.ylabel("Brightness")
    plt.grid(True)
    plt.legend()

    out_path = f"plots/{video_name}_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved:", out_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_curves <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print("❌ File not found:", video_path)
        sys.exit(1)

    plot_curves(video_path)


if __name__ == "__main__":
    main()
