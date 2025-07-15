import os
import cv2
import random
import numpy as np
import shutil
import time
from tqdm import tqdm
from skimage.util import random_noise

random.seed(0)

def uniform_interval_sample(images, sample_count):
    total = len(images)
    if sample_count >= total:
        return images
    # Take sample_count positions from 0 to total-1 using an arithmetic index
    step_indices = [int(i * (total - 1) / (sample_count - 1)) for i in range(sample_count)]
    return [images[idx] for idx in step_indices]
# Add noise function
def add_noise(image, method, level):
    noisy_image = None
    if method == "gaussian":
        noise = random_noise(image, mode='gaussian', var=level)
        noisy_image = (255 * noise).astype(np.uint8)
    elif method == "poisson":
        noise = random_noise(image, mode='poisson', clip=True, seed=None) * level
        noisy_image = (255 * noise).astype(np.uint8)
    elif method == "salt_pepper":
        noise = random_noise(image, mode='s&p', amount=level)
        noisy_image = (255 * noise).astype(np.uint8)

    return noisy_image


input_path = r"HA419-IonogramSet-202505\2002_2015_classification"
output_path = r"HA419-IonogramSet-202505\HA_80_20_uniform"
image_count_threshold = 30000
noise_levels = {
    'gaussian': [0.05, 0.10, 0.15],
    'poisson': [1, 2, 4],
    'salt_pepper': [0.1, 0.15, 0.2]
}

for class_folder in os.listdir(input_path):
    print()
    print("Processing images of the {} category".format(class_folder))
    folder_path = os.path.join(input_path, class_folder)
    folder_images = os.listdir(folder_path)
    image_num = len(folder_images)

    # Create the output path and its subfolders
    class_folder_output_path = os.path.join(output_path, class_folder)
    if not os.path.exists(class_folder_output_path):
        os.makedirs(class_folder_output_path)

    # If the number of original images exceeds the threshold
    if image_num >= image_count_threshold:
        # Randomly select 20,000 pictures
        folder_images = uniform_interval_sample(folder_images, image_count_threshold)

        # Save randomly selected pictures in a loop
        for filename in tqdm(folder_images, desc="Filter and save the pictures"):
            image_path = os.path.join(folder_path, filename)
            new_image_path = os.path.join(class_folder_output_path, filename)
            shutil.copyfile(image_path, new_image_path)

    else:
        # Save the existing pictures in a loop
        for filename in tqdm(folder_images, desc="Save the original picture first"):
            image_path = os.path.join(folder_path, filename)
            new_image_path = os.path.join(class_folder_output_path, filename)
            shutil.copyfile(image_path, new_image_path)

        # Randomly add noise in a loop
        while image_num < image_count_threshold:
            for filename in folder_images:
                # Generate random variables
                random_index = random.randint(0, 8)

                # Determine the method of adding noise based on random variables
                if random_index < 3:
                    method = "gaussian"
                    level = noise_levels["gaussian"][random_index]
                elif 3 <= random_index < 6:
                    method = "poisson"
                    level = noise_levels["poisson"][random_index - 3]
                else:
                    method = "salt_pepper"
                    level = noise_levels["salt_pepper"][random_index - 6]

                print("\rAdd noise and save the picture: {}/{}".format(image_num, image_count_threshold), end="")
                # This line of judgment cannot be deleted either
                if image_num >= image_count_threshold:
                    break

                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                # Determine the number of column and block noises to be added, all of which are random numbers within the range
                col_num = random.randint(0, 3)
                rec_num = random.randint(1, 3) if col_num == 0 else random.randint(0, 3)

                h, w, _ = image.shape

                # Step 1: Select a certain consecutive 1-5 column segment of the image to add noise
                for i in range(col_num):
                    start_col = random.randint(0, w-5)
                    end_col = start_col + random.randint(1, 5)
                    image[:, start_col:end_col] = add_noise(image[:, start_col:end_col], method, level)

                # Step 2: Add noise to a parallelogram area of a certain a*b in the image. The parallelogram area is inclined at a 45-degree Angle towards the upper right corner
                rec_b = random.randint(35, 45)  # bottom
                rec_h = random.randint(18, 25)  # high

                for i in range(rec_num):
                    start_col = random.randint(0, w - rec_h)
                    start_row = random.randint(rec_h - 1, h - rec_b - rec_h)  # The minimum value that can be taken is rec_h-1; Subtract the base and the extended part of the parallelogram
                    for j in range(rec_h):
                        noise_area = image[start_row:start_row + rec_b + rec_h, start_col:start_col+1]
                        image[start_row:start_row+rec_b+rec_h, start_col:start_col+1] = add_noise(noise_area, method,
                                                                                                  level)
                        start_col += 1
                        start_row -= 1

                new_filename = '{}_{}_{}_{}.png'.format(filename.split('.')[0], method, level, image_num+1)  # Prevent name duplication!
                new_image_path = os.path.join(class_folder_output_path, new_filename)
                cv2.imwrite(new_image_path, image)

                image_num += 1

    time.sleep(0.5)

print("\nProcessing completed")
