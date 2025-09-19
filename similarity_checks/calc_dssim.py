from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sys
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def compare_images(imageA, imageB):
    # Load the two PNG images (convert to grayscale for SSIM)
    image1 = imread(imageA, as_gray=True)
    image2 = imread(imageB, as_gray=True)

    # Compute SSIM
    score, diff = ssim(image1, image2, full=True, data_range=1.0)
    print(f"Image 1: {imageA}")
    print(f"Image 2: {imageB}")
    # print(f"Difference Image: {diff}")
    print(f"SSIM Score: {score:.4f}")
    # psnr_score = peak_signal_noise_ratio(image1, image2, data_range=1.0)
    # print(f"PSNR Score: {psnr_score:.4f} dB")

    # (Optional) Visualize the difference image
    # plt.imshow(diff, cmap='hot')
    # plt.title('SSIM Difference Image')
    # plt.axis('off')
    # plt.savefig('ssim_difference_image.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # Create a single figure
    # # plt.figure(figsize=(8, 8))
    # plt.figure(figsize=(image1.shape[1]/100, image1.shape[0]/100))
    # # Display original image with difference overlay
    # plt.imshow(imread(imageA))
    # plt.imshow(diff, cmap='hot', alpha=0.5)  # Alpha controls transparency
    # plt.axis('off')
    
    # plt.tight_layout()
    # output_path = imageA.rsplit('.png', 1)[0] + '_diff.png'
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # # plt.show()

def compare_images_by_dataset(dataset_id):
    base_path = '/mnt/d/LDAV25Data/traveler-benchmarks/' + dataset_id
    alg = ['db_duck_raw', 'summed_area_table', 'agglomerative_clustering', 'eseman_kdt', 'db_duck_min_max', 'db_duck_sketch']
    query = ['window', 'attribute', 'cond']
    hrd = 1
    for alg_name in alg:
        file1 = f"{base_path}/{dataset_id}_{query[0]}_{alg_name}_{hrd}_gantt.png"
        file2 = f"{base_path}/{dataset_id}_{query[0]}_{alg[0]}_{hrd}_gantt.png"
    
        # Compare the two images
        compare_images(file1, file2)
        print("----------------------------------")

def main():
    DGEM_ID = 'a9bd20ca-c4f2-4b54-8c49-b968ae7e78be'#'589ca754-ef75-426c-8d51-841cc61dc84a'
    KMEANS_ID = '8b3289c9-a740-4091-a56d-e4d55af526b5'#'faf17535-2f66-4621-995f-49c7dbd84e8b'
    LULESH_ID = '772c7330-d4eb-485b-866a-3b315063f9af'

    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        compare_images(file1, file2)
    elif len(sys.argv) == 2:
        dataset_id = sys.argv[1]
        compare_images_by_dataset(dataset_id)
    else:
        print("Usage: python calc_dssim.py <file1> <file2>")
        print("Usage: python calc_dssim.py <dataset id>")
        return
if __name__ == '__main__':
    main()
    # Create a custom colormap highlighting high/low values

    # # Create data points for visualization
    # x = np.linspace(0, 10, 100)
    # y = np.linspace(0, 10, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = np.sin(X) * np.cos(Y)

    # # Define custom colormap colors
    # n_bins = 100
    # colors = plt.cm.hot(np.linspace(0, 1, n_bins))
    # custom_hot = LinearSegmentedColormap.from_list("custom_hot", colors, N=n_bins)

    # # Plot
    # plt.figure(figsize=(10, 8))
    # plt.imshow(Z, cmap=custom_hot)
    
    # # Add horizontal colorbar at the bottom
    # cbar = plt.colorbar(orientation='horizontal')
    # cbar.set_ticks([Z.min(), Z.max()])
    # cbar.set_ticklabels(['Low', 'High'])
    
    # plt.title('Custom Hot Colormap')
    # plt.show()