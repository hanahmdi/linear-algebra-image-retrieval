
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

class FaceRecognitionSystem:
    def __init__(self):
        print("--- Initializing Advanced System ---")
        try:
            # Load Data
            data = fetch_olivetti_faces()
            self.faces = data.images
            self.n_samples, self.h, self.w = self.faces.shape
            print(f"-> Dataset Loaded: {self.n_samples} images")
        except:
            print("Error: Could not download dataset.")
            exit()

    def run_full_demo(self):
        print("\n=== STARTING HARD MODE DEMO ===\n")
        self.step_1_find_face_in_crowd_hard()
        self.step_2_svd_compression()
        self.step_3_pca_recognition()
        print("\n=== DEMO COMPLETED SUCCESSFULLY ===")

    def step_1_find_face_in_crowd_hard(self):
        print(">> Step 1: Template Matching (Hard Mode)...")
        
        # 1. Create a Very Busy Crowd (8 rows x 8 columns = 64 faces)
        print("   -> Generating dense crowd image (64 faces)...")
        rows = []
        for _ in range(8):
            indices = np.random.choice(self.n_samples, 8, replace=False)
            row_imgs = np.hstack([self.faces[i] for i in indices])
            rows.append(row_imgs)
        
        crowd_image = np.vstack(rows)
        
        # 2. Define the Target Location (Hidden somewhere in the crowd)
        target_y, target_x = 200, 300 # Random coordinates inside the big image
        h, w = self.h, self.w
        
        # Extract the CLEAN target first
        clean_target = crowd_image[target_y:target_y+h, target_x:target_x+w]
        
        # 3. Make the Template IMPERFECT (Add Noise/Distortion)
        # This simulates a bad camera or an old photo
        print("   -> Corrupting the template (adding noise) to make matching harder...")
        noise = np.random.normal(0, 0.15, clean_target.shape) # Add 15% random noise
        noisy_template = clean_target + noise
        
        # Clip values to stay valid image range
        noisy_template = np.clip(noisy_template, 0, 1)

        print("   -> Scanning large image with SSD Algorithm...")
        
        # 4. Search Algorithm (SSD)
        H, W = crowd_image.shape
        step = 2
        min_diff = float('inf')
        best_loc = (0, 0)
        
        for y in range(0, H - h, step):
            for x in range(0, W - w, step):
                patch = crowd_image[y:y+h, x:x+w]
                # Compare the NOISY template with the CLEAN crowd patch
                diff = np.sum((patch - noisy_template)**2)
                
                if diff < min_diff:
                    min_diff = diff
                    best_loc = (y, x)
        
        found_y, found_x = best_loc
        
        # 5. Visualization
        plt.figure(figsize=(12, 8))
        
        # Show the Noisy/Imperfect Template
        plt.subplot(1, 2, 1)
        plt.imshow(noisy_template, cmap='gray')
        plt.title("Input Template\n(Noisy & Low Quality)", fontsize=14, color='red', fontweight='bold')
        plt.axis('off')
        
        # Show Result
        plt.subplot(1, 2, 2)
        plt.imshow(crowd_image, cmap='gray')
        plt.title(f"Crowd Search Result\n(Found despite noise)", fontsize=14, color='green')
        
        # Red Box
        rect = plt.Rectangle((found_x, found_y), w, h, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(found_x, found_y-10, "MATCH FOUND", color='red', fontsize=10, fontweight='bold', backgroundcolor='white')
        
        plt.tight_layout()
        plt.show()

    def step_2_svd_compression(self):
        print("\n>> Step 2: SVD Compression...")
        img = self.faces[10]
        U, S, Vt = np.linalg.svd(img, full_matrices=False)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        for i, k in enumerate([5, 15]):
            S_k = np.diag(S[:k])
            compressed = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
            plt.subplot(1, 3, i+2)
            plt.imshow(compressed, cmap='gray')
            plt.title(f"Compressed (k={k})")
            plt.axis('off')
            
        plt.show()

    def step_3_pca_recognition(self):
        print("\n>> Step 3: PCA Face Recognition...")
        n_components = 50
        X = self.faces.reshape(self.n_samples, -1)
        
        pca = PCA(n_components=n_components, whiten=True).fit(X)
        X_pca = pca.transform(X)
        
        query_idx = 45
        query_vec = X_pca[query_idx].reshape(1, -1)
        
        dists = np.linalg.norm(X_pca - query_vec, axis=1)
        dists[query_idx] = np.inf 
        match_idx = np.argmin(dists)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.faces[query_idx], cmap='gray')
        plt.title("Unknown Person")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.faces[match_idx], cmap='gray')
        plt.title(f"Identified as ID: {match_idx}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    app = FaceRecognitionSystem()
    app.run_full_demo()
