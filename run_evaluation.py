"""
Run model evaluation and display the confusion matrix
"""
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def run_evaluation():
    """Run the evaluation script and display results"""
    print("Running model evaluation...")
    
    # Make sure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Run the evaluation script
    cmd = ["python", "-m", "src.evaluate", 
           "--model", "artifacts/model.keras", 
           "--features", "mfcc", 
           "--with-deltas"]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print("Error running evaluation:")
        print(process.stderr)
        return False
    
    # Print the output
    print("\nEvaluation Results:")
    print("=" * 50)
    print(process.stdout)
    
    # Display classification report
    if os.path.exists("artifacts/classification_report_eval.txt"):
        print("\nClassification Report:")
        print("=" * 50)
        with open("artifacts/classification_report_eval.txt", "r") as f:
            print(f.read())
    
    # Display confusion matrix image
    if os.path.exists("artifacts/confusion_matrix_eval.png"):
        print("\nConfusion Matrix saved to artifacts/confusion_matrix_eval.png")
        
        # If running in interactive environment, show the image
        try:
            plt.figure(figsize=(10, 8))
            img = mpimg.imread("artifacts/confusion_matrix_eval.png")
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            print("Confusion matrix displayed. Close the image window to continue.")
        except Exception as e:
            print(f"Couldn't display image: {e}")
            print("Please open the image file manually.")
    
    return True

if __name__ == "__main__":
    run_evaluation()
