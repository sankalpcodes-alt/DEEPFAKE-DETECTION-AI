import os

from utils.image_inference import predict_image


def run_baseline():
    print("Training Baseline Model")
    os.system("python training/train_baseline.py")


def run_proposed():
    print("Training Proposed Model")
    os.system("python training/train_proposed.py")


def run_evaluation():
    print("Evaluating Models")
    os.system("python evaluation/evaluate_models.py")


def run_all():
    run_baseline()
    run_proposed()
    run_evaluation()


def run_image_check():
    image_path = input("Enter image path: ").strip().strip('"')

    if not image_path:
        print("No image path provided.")
        return

    if not os.path.exists(image_path):
        print("Image file not found.")
        return

    try:
        result = predict_image(image_path)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return

    print("\nImage Prediction")
    print("Baseline Model Result:", result["baseline_label"])
    print("Baseline Real Probability:", f"{result['baseline_real_probability'] * 100:.2f}%")
    print("Baseline Fake Probability:", f"{result['baseline_fake_probability'] * 100:.2f}%")
    print("Proposed Model Result:", result["proposed_label"])
    print("Proposed Real Probability:", f"{result['proposed_real_probability'] * 100:.2f}%")
    print("Proposed Fake Probability:", f"{result['proposed_fake_probability'] * 100:.2f}%")


while True:
    print("\nChoose an option:")
    print("1. Train Baseline Model")
    print("2. Train Proposed Model")
    print("3. Evaluate Models")
    print("4. Run All")
    print("5. Check Single Image")
    print("6. Quit")

    choice = input("Enter choice (1-6): ").strip()

    if choice == "1":
        run_baseline()
    elif choice == "2":
        run_proposed()
    elif choice == "3":
        run_evaluation()
    elif choice == "4":
        run_all()
    elif choice == "5":
        run_image_check()
    elif choice == "6":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please enter 1 to 6.")
