from inference import load_model, start_interface

SYSTEM_NAME = "Entropy-Driven Uncertainty-Aware Diagnostic System"
SYSTEM_VERSION = "v1.0"

MODEL_PATH = "model.pt"


def main():
    print("=" * 60)
    print(SYSTEM_NAME)
    print(f"Version : {SYSTEM_VERSION}")
    print("Purpose : Uncertainty-aware decision support with explicit ignorance")
    print("=" * 60)

    model = load_model(MODEL_PATH)
    start_interface(model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
