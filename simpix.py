import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from numba import jit

@jit(nopython=True)
def compute_gradient_field(src_array, tgt_array):
    """Compute gradient for each pixel position."""
    height, width = src_array.shape[:2]
    gradients = np.zeros((height, width), dtype=np.int64)  # Use int64 to prevent overflow
    
    for i in range(height):
        for j in range(width):
            # Use int64 for all intermediate calculations
            dred = np.int64(src_array[i, j, 0]) - np.int64(tgt_array[i, j, 0])
            dgreen = np.int64(src_array[i, j, 1]) - np.int64(tgt_array[i, j, 1])
            dblue = np.int64(src_array[i, j, 2]) - np.int64(tgt_array[i, j, 2])
            gradients[i, j] = np.abs(dred) + np.abs(dgreen) + np.abs(dblue)
    
    return gradients

@jit(nopython=True)
def gradient_at_position(src_pixel, tgt_pixel):
    """Calculate gradient between two pixels. Returns int64 to prevent overflow."""
    # Cast to int64 immediately to prevent overflow in differences
    dred = np.int64(src_pixel[0]) - np.int64(tgt_pixel[0])
    dgreen = np.int64(src_pixel[1]) - np.int64(tgt_pixel[1])
    dblue = np.int64(src_pixel[2]) - np.int64(tgt_pixel[2])
    return np.abs(dred) + np.abs(dgreen) + np.abs(dblue)

@jit(nopython=True)
def metropolis_optimized(src_array, tgt_array, gradients, T, nsweeps):
    """
    Optimized Metropolis algorithm using cached gradients.
    Modifies src_array and gradients IN-PLACE.
    """
    height, width = src_array.shape[:2]
    accepted = 0
    improvements = 0
    neutral_moves = 0
    rejected_uphill = 0
    
    for n in range(nsweeps):
        # Randomly select two different pixels
        i1 = np.random.randint(0, height)
        j1 = np.random.randint(0, width)
        i2 = np.random.randint(0, height)
        j2 = np.random.randint(0, width)
        
        # Make sure they're different pixels
        if (i1 == i2) and (j1 == j2):
            continue
        
        # Current gradients at both positions (already int64)
        grad_before_1 = gradients[i1, j1]
        grad_before_2 = gradients[i2, j2]
        
        # Calculate what gradients would be after swap (returns int64)
        grad_after_1 = gradient_at_position(src_array[i2, j2], tgt_array[i1, j1])
        grad_after_2 = gradient_at_position(src_array[i1, j1], tgt_array[i2, j2])
        
        # Change in total gradient - all int64 so no overflow
        # Compute as: (after1 - before1) + (after2 - before2) to be extra safe
        delta_grad = (grad_after_1 - grad_before_1) + (grad_after_2 - grad_before_2)
        
        # Metropolis acceptance criterion
        accept = False
        if delta_grad < 0:
            # Improvement - always accept
            accept = True
            improvements += 1
        elif delta_grad == 0:
            # Neutral move - always accept (no energy change)
            accept = True
            neutral_moves += 1
        elif T > 0:
            # Uphill move - accept with probability exp(-delta/T)
            if np.random.random() < np.exp(-float(delta_grad) / T):
                accept = True
            else:
                rejected_uphill += 1
        else:
            # T == 0, reject all uphill moves
            rejected_uphill += 1
        
        if accept:
            # Perform the swap IN-PLACE
            temp_pixel = src_array[i1, j1].copy()
            src_array[i1, j1] = src_array[i2, j2]
            src_array[i2, j2] = temp_pixel
            
            # Update cached gradients
            gradients[i1, j1] = grad_after_1
            gradients[i2, j2] = grad_after_2
            
            accepted += 1
    
    return accepted, improvements, neutral_moves, rejected_uphill

def simulated_annealing(src_array, tgt_array, T_init=1000, T_min=0.1, 
                       alpha=0.99, nsweeps=1000, greedy_iterations=10):
    """
    Simulated annealing for pixel rearrangement.
    src_array is modified IN-PLACE.
    """
    # Compute initial gradient field
    print("Computing initial gradient field...")
    gradients = compute_gradient_field(src_array, tgt_array)
    current_total_grad = np.sum(gradients)
    
    print(f"Starting gradient: {current_total_grad}")
    
    T = T_init
    temp_iteration = 0
    
    # Phase 1: Simulated Annealing
    print("\n=== Phase 1: Simulated Annealing ===")
    while T > T_min:
        # Run Metropolis at current temperature
        accepted, improvements, neutral_moves, rejected_uphill = metropolis_optimized(
            src_array, tgt_array, gradients, T, nsweeps)
        
        # Cool down
        T *= alpha
        temp_iteration += 1
        
        # Progress indicator every 50 temperature steps
        if temp_iteration % 50 == 0:
            current_total_grad = np.sum(gradients)
            acceptance_rate = accepted / nsweeps * 100
            improvement_rate = improvements / nsweeps * 100
            neutral_rate = neutral_moves / nsweeps * 100
            uphill_acceptance = (accepted - improvements - neutral_moves) / nsweeps * 100
            print(f"Iteration {temp_iteration}, T={T:.4f}, "
                  f"accepted={accepted}/{nsweeps} ({acceptance_rate:.1f}%), "
                  f"improvements={improvements} ({improvement_rate:.1f}%), "
                  f"neutral={neutral_moves} ({neutral_rate:.1f}%), "
                  f"uphill_accepted={uphill_acceptance:.2f}%, "
                  f"gradient={current_total_grad:.0f}")
    
    grad_after_annealing = np.sum(gradients)
    print(f"Gradient after annealing: {grad_after_annealing}")
    
    # Phase 2: Greedy Refinement (T=0, only accept improvements)
    print(f"\n=== Phase 2: Greedy Refinement ({greedy_iterations} iterations) ===")
    for greedy_iter in range(greedy_iterations):
        accepted, improvements, neutral_moves, rejected_uphill = metropolis_optimized(
            src_array, tgt_array, gradients, 0.0, nsweeps)
        
        current_total_grad = np.sum(gradients)
        acceptance_rate = accepted / nsweeps * 100
        improvement_rate = improvements / nsweeps * 100
        print(f"Greedy iteration {greedy_iter + 1}/{greedy_iterations}, "
              f"improvements={improvements} ({improvement_rate:.3f}%), "
              f"gradient={current_total_grad:.0f}")
        
        # Stop early if no improvements found
        if improvements == 0:
            print(f"No improvements found, stopping greedy refinement early.")
            break
    
    final_grad = np.sum(gradients)
    improvement = grad_after_annealing - final_grad
    print(f"\nFinal gradient: {final_grad}")
    print(f"Improvement from greedy phase: {improvement:.0f} ({improvement/grad_after_annealing*100:.2f}%)")
    
    return src_array

def main():
    if len(sys.argv) < 3:
        print("Usage: simapix_start image1 image2 <output=out.png>")
        return

    fsrc = sys.argv[1]
    ftgt = sys.argv[2]
    fout = sys.argv[3] if len(sys.argv) > 3 else "out.png"
    
    print(f"Reading images: source={fsrc} target={ftgt}")
    print(f"Output={fout}")

    # Open images
    src = Image.open(fsrc)
    tgt = Image.open(ftgt)

    print(f"Source size: {src.width} x {src.height}")
    print(f"Target size: {tgt.width} x {tgt.height}")

    # Handle different image sizes
    if src.size != tgt.size:
        print(f"\nImages have different sizes!")
        print(f"Resizing source image to match target size...")
        
        # Resize source to match target
        src = src.resize((tgt.width, tgt.height), Image.LANCZOS)
        print(f"Source resized to: {src.width} x {src.height}")

    print(f"Working with geometry: {src.width} x {src.height}")

    # Convert images to numpy arrays
    src_array = np.array(src)
    tgt_array = np.array(tgt)
    out_array = np.copy(src_array)
    
    # Diagnostic: Check if images are identical
    if np.array_equal(src_array, tgt_array):
        print("WARNING: Source and target images are identical!")
    
    # Run simulated annealing with optimized parameters
    out_array = simulated_annealing(out_array, tgt_array, 
                                    T_init=500, 
                                    T_min=0.00001,  # Much lower minimum temperature
                                    alpha=0.92,      # Slightly faster cooling
                                    nsweeps=1000000,
                                    greedy_iterations=20)  # 20 rounds of pure greedy optimization

    # Create a collage of images
    fig, axs = plt.subplots(2, 2, figsize=(8, 4))
    
    axs[0,0].imshow(src)
    axs[0,0].set_title('Source Image')
    axs[0,0].axis('off')
    
    axs[0,1].imshow(tgt)
    axs[0,1].set_title('Target Image')
    axs[0,1].axis('off')
    
    axs[1, 0].imshow(out_array)
    axs[1, 0].set_title('Output Image')
    axs[1, 0].axis('off')

    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("collage.png")
    
    # Save the new image
    out_image = Image.fromarray(out_array)
    out_image.save(fout)

    print("Close image window to exit")
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")