# There is probably a better way, but here is my way, at least for now

import numpy as np
import trimesh

# Function to create a small block (cube)
def create_small_block(base_size, height, position):
    # Create a small cube using trimesh
    block = trimesh.creation.box(extents=[base_size, base_size, height])
    block.apply_translation(position)
    return block

# Function to create the large composite pyramid
def create_large_pyramid(base_size, small_block_base_size, small_block_height, overlap_factor):
    large_pyramid = []
    
    # Loop until the base size is reduced to one block
    current_base_size = base_size
    current_height = 0
    
    while current_base_size >= small_block_base_size:
        # Adjust small block dimensions to account for overlap factor
        adjusted_block_size = small_block_base_size * (1 - overlap_factor)
        adjusted_block_height = small_block_height * (1 - overlap_factor)

        # Calculate the number of blocks per side for the current layer
        num_blocks_per_side = max(1, int(current_base_size / adjusted_block_size))
        
        # Z height for this layer
        z = current_height

        # Loop through the blocks in this layer and position them
        for i in range(num_blocks_per_side):
            for j in range(num_blocks_per_side):
                # X, Y positions with some offset for overlapping effect
                x = -current_base_size / 2 + i * adjusted_block_size
                y = -current_base_size / 2 + j * adjusted_block_size
                position = [x, y, z]
                
                # Create a small block and add it to the pyramid
                small_block = create_small_block(adjusted_block_size, adjusted_block_height, position)
                large_pyramid.append(small_block)

        # Reduce the base size and increase the current height for the next layer
        current_base_size -= adjusted_block_size
        current_height += adjusted_block_height

        # Stop when there's only one block in the top layer
        if num_blocks_per_side == 1:
            break

    # Combine all the small blocks into one mesh
    large_pyramid_mesh = trimesh.util.concatenate(large_pyramid)
    large_pyramid_mesh.apply_translation([0, 0, -large_pyramid_mesh.bounds[0][2]])  # Adjust to sit on the plane
    return large_pyramid_mesh

# Parameters
base_size = 120  # Size of the base of the large pyramid
small_block_base_size = 3  # Base size of each small block
small_block_height = 1.5  # Height of each small block
overlap_factor = 0.3  # Overlap between layers

# Create the large composite pyramid
large_pyramid_mesh = create_large_pyramid(base_size, small_block_base_size, small_block_height, overlap_factor)

# Export the result as STL file
large_pyramid_mesh.export('pyramid_blocks.stl')

print("Large composite pyramid saved as 'composite_pyramid_blocks_fixed.stl'.")

