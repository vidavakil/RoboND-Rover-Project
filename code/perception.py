import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(162, 162, 162)): # was 160
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Locate the rock
def find_rock(img, rgb_thresh=(110, 110, 50)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    yellow_thresh = (img[:,:,0] > rgb_thresh[0]) \
        & (img[:,:,1] > rgb_thresh[1]) \
        & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[yellow_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at
    # the center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    mask = cv2.warpPerspective(np.ones([img.shape[0], img.shape[1], 1]), M,
        (img.shape[1], img.shape[0])) # mask out everything that is not visible

    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    image = Rover.img
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                          [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                          [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                          [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                          ])
    
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    navigable, mask = perspect_transform(color_thresh(Rover.img), source, destination)
    found_rock, mask = perspect_transform(find_rock(Rover.img), source, destination)
    obstacles = mask - navigable
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacles * 255
    Rover.vision_image[:,:,2] = navigable * 255
    
    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    world_map_height = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    
    if (Rover.step < 10):
        Rover.previous_pos[Rover.step, :] = Rover.pos[:]
    else:
        Rover.previous_pos[0:9, :] = Rover.previous_pos[1:, :]
        Rover.previous_pos[9, :] = Rover.pos[:]
    Rover.step += 1

    # To increase Fidelity, check the pitch and roll angles
    valid_angles = 0
    if (Rover.pitch < 1.0 or Rover.pitch > 364.0) \
        and (Rover.roll < 1.0 or Rover.roll > 364.0):
        valid_angles = 1

    # Notice that y and x are swapped in the worldmap!
    xpix, ypix = rover_coords(obstacles)
    obstacles_x_pix_world, obstacles_y_pix_world = pix_to_world(
        xpix, ypix, xpos, ypos, yaw, world_map_height, scale)
    if valid_angles:
        Rover.worldmap[obstacles_y_pix_world, obstacles_x_pix_world, 0] += 1 # * Rover.vel
    
    if found_rock.any():
        xpix, ypix = rover_coords(found_rock)
        found_rock_x_pix_world, found_rock_y_pix_world = pix_to_world(
            xpix, ypix, xpos, ypos, yaw, world_map_height, scale)
        
        rock_dist, rock_angles = to_polar_coords(xpix, ypix)
        rock_idx = np.argmin(rock_dist)
        rock_xcenter = found_rock_x_pix_world[rock_idx]
        rock_ycenter = found_rock_y_pix_world[rock_idx]
        
        if valid_angles:
            Rover.worldmap[rock_ycenter, rock_xcenter, 1] = 255
        Rover.vision_image[:,:,1] = found_rock * 255
    
        Rover.rock_pos[0] = rock_xcenter
        Rover.rock_pos[1] = rock_ycenter
        Rover.rock_angle = rock_angles[rock_idx]
        Rover.found_rock = 1
    
        print ('Robot position =', Rover.pos, 'Rock Position', Rover.rock_pos)
    else:
        Rover.vision_image[:,:,1] = 0
        Rover.found_rock = 0

    xpix, ypix = rover_coords(navigable)
    navigable_x_pix_world, navigable_y_pix_world = pix_to_world(
        xpix, ypix, xpos, ypos, yaw, world_map_height, scale)
    if valid_angles:
        Rover.worldmap[navigable_y_pix_world, navigable_x_pix_world, 2] += 20 # * Rover.vel

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles
    dist, angles = to_polar_coords(xpix, ypix)

    # For steering, don't look too far away. Only consider adjacent navigable
    # terrain. This will help manuvering around corners, without bumping to them.
    indices = np.argwhere(dist < 50)
    Rover.nav_angles = angles
    Rover.adjacent_angles = angles[indices]
    Rover.nav_dists = dist

    return Rover
