
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

MAIN_IMAGE_FILE = 'main.jpg.png'   

TEMPLATE_IMAGE_FILE = 'template.jpg.png' 

def run_generalizability_test():
    print("--- Generalizability Test: Custom Object Detection ---")
    
    # 1. Check if files exist
    if not os.path.exists(MAIN_IMAGE_FILE) or not os.path.exists(TEMPLATE_IMAGE_FILE):
        print(f"Error: Images not found!")
        print(f"Looking for: '{MAIN_IMAGE_FILE}' and '{TEMPLATE_IMAGE_FILE}'")
        return

    # 2. Load Images
    img_rgb = cv2.imread(MAIN_IMAGE_FILE)
    template = cv2.imread(TEMPLATE_IMAGE_FILE)
    
    if img_rgb is None or template is None:
        print("Error: Files exist but OpenCV could not read them.")
        return

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions of the template
    h, w = template_gray.shape

    # 3. Perform Linear Algebra Correlation (Template Matching)
    print(f"Scanning '{MAIN_IMAGE_FILE}' for the target pattern...")
    
    # --- FIX IS HERE: usage of cv2.TM_CCOEFF_NORMED ---
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find the location with the highest correlation score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Define the bounding box coordinates
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # 4. Visualization
    result_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) 
    cv2.rectangle(result_image, top_left, bottom_right, (255, 0, 0), 4) 
    
    # Add label with confidence score
    confidence_text = f"Match: {max_val*100:.1f}%"
    cv2.putText(result_image, confidence_text, (top_left[0], top_left[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Left: The Target
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title("Target Template", fontsize=14)
    plt.axis('off')
    
    # Right: The Result
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title(f"Detection Result (Score: {max_val:.2f})", fontsize=14)
    plt.axis('off')
    
    plt.suptitle("Generalizability Test: Custom Data", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("Process finished. Check the popup window.")

if __name__ == "__main__":
    run_generalizability_test()
