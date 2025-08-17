"""
Run script to train the improved audio classification model
"""
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the improved audio classification pipeline")
    parser.add_argument("--debug", action="store_true", help="Run debug visualizations first")
    parser.add_argument("--features", choices=["mfcc", "logmel"], default="mfcc", 
                      help="Feature type (default: mfcc)")
    parser.add_argument("--with-deltas", action="store_true", 
                      help="Include delta and acceleration features")
    parser.add_argument("--epochs", type=int, default=15, 
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                      help="Batch size for training")
    parser.add_argument("--realtime", action="store_true",
                      help="Optimize model for real-time inference")
    args = parser.parse_args()

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    print("="*60)
    print("IMPROVED SPOKEN DIGIT CLASSIFICATION PIPELINE")
    print("="*60)
    
    # First run debugging and visualization if requested
    if args.debug:
        print("\nRunning data visualization and debugging...")
        subprocess.run(["python", "debug_visualize.py"], check=True)
        print("\nRunning model debugging script...")
        subprocess.run(["python", "debug_model.py"], check=True)
    
    # Build training command
    train_cmd = ["python", "-m", "src.train"]
    train_cmd.extend(["--features", args.features])
    if args.with_deltas:
        train_cmd.append("--with-deltas")
    train_cmd.extend(["--epochs", str(args.epochs)])
    train_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.realtime:
        train_cmd.append("--realtime")
    
    print("\nTraining model with command:")
    print(" ".join(train_cmd))
    print("-"*60)
    
    # Run training
    try:
        subprocess.run(train_cmd, check=True)
        print("\nTraining complete!")
        print("\nEvaluating model...")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        print("Please check the error message above and fix any issues.")
        return
    
    # Run evaluation
    eval_cmd = ["python", "-m", "src.evaluate", 
                "--model", "artifacts/model.keras",
                "--features", args.features]
    if args.with_deltas:
        eval_cmd.append("--with-deltas")
    
    subprocess.run(eval_cmd, check=True)
    
    print("\nModel evaluation complete!")
    print("Check artifacts directory for results.")
    
    # Ask if user wants to run real-time inference
    if args.realtime:
        try:
            use_gui = input("\nDo you want to use GUI visualization for real-time inference? (y/n): ").strip().lower()
            
            if use_gui.startswith('y'):
                print("\nStarting real-time inference with GUI visualization...")
                subprocess.run(["python", "realtime_inference.py", 
                               "--features", args.features,
                               "--model", "artifacts/model.keras"] + 
                               (["--with-deltas"] if args.with_deltas else []))
            else:
                print("\nStarting command-line real-time inference...")
                subprocess.run(["python", "realtime_cli.py",
                               "--features", args.features,
                               "--model", "artifacts/model.keras"] + 
                               (["--with-deltas"] if args.with_deltas else []))
        except KeyboardInterrupt:
            print("\nStopping real-time inference...")
    else:
        print("\nYou can run real-time inference using one of these commands:")
        print(f"  python realtime_inference.py --features {args.features}" + 
              (" --with-deltas" if args.with_deltas else "") + 
              " # for GUI visualization")
        print(f"  python realtime_cli.py --features {args.features}" + 
              (" --with-deltas" if args.with_deltas else "") + 
              " # for command-line visualization")
    
    print("="*60)

if __name__ == "__main__":
    main()
