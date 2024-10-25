from PIL import Image
import os

# Define the image folder and output GIF path
image_folder = 'MOTS_sequence_demo/0001_MOTS_combined'
gif_path = 'MOTS_sequence_demo/0001_MOTS_combined/labeled_sequence.gif'

# Get list of images and sort them in order
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Load images into a list
frames = [Image.open(os.path.join(image_folder, image)) for image in images]

# Save as GIF
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
print("GIF saved at:", gif_path)
