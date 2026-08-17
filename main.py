import argparse
import sys
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Road Surface Crack Detection Project Entry Point")
    parser.add_argument('--setup', action='store_true', help='Download and preprocess dataset')
    parser.add_argument('--jupyter', action='store_true', help='Launch Jupyter Notebook to run the project')
    parser.add_argument('--infer', type=str, metavar='INPUT_PATH', help='Run crack detection inference on an image or video file')
    parser.add_argument('--output', type=str, default=None, help='Output path for inference result')
    
    args = parser.parse_args()
    
    if args.setup:
        print("Downloading and extracting dataset...")
        subprocess.run([sys.executable, os.path.join('support', 'crack', 'download_data.py'), '--extract'], check=True)
        print("Preprocessing dataset...")
        subprocess.run([sys.executable, os.path.join('support', 'crack', 'preprocess.py')], check=True)
        print("Setup complete!")
    elif args.jupyter:
        print("Starting Jupyter Notebook...")
        subprocess.run(['jupyter', 'notebook'], check=True)
    elif args.infer:
        from support.shared.inference import CrackPredictor
        predictor = CrackPredictor()
        predictor.process(input_path=args.infer, output_path=args.output)
    else:
        print("Welcome to the Road Surface Crack Detection Project!")
        print("To run the project, the primary interface is through the provided Jupyter Notebooks.")
        print("\nAvailable commands:")
        print("  python main.py --setup                       # Downloads and preprocesses the datasets")
        print("  python main.py --infer <image_or_video_path> # Runs crack detection on image or video")
        print("  python main.py --jupyter                     # Launches Jupyter Notebook interface")
        print("\nIf you want to train the model, open 'notebooks/02_train_crack_unet.ipynb'.")
        print("For interactive image/video inference, open 'crack_inference.ipynb'.\n")
        parser.print_help()

if __name__ == "__main__":
    main()
