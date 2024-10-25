import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def Combine_MOTS_image(rgb_path, MOTS_mask_path):
    rgb_image = cv2.imread(rgb_path)[:, :, ::-1]
    MOTS_mask = np.array(Image.open(MOTS_mask_path))
    obj_ids = np.unique(MOTS_mask)

    color_mask = np.zeros_like(rgb_image)

    # Assign a color to each object based on class_id and object_instance_id
    for obj_id in obj_ids:
        if obj_id == 0:
            continue  # Skip background

        # Compute class_id and obj_instance_id
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000

        if class_id not in [1, 2]:
            continue # Skip 10000?

        np.random.seed(obj_instance_id)  # Ensure consistent color for the same instance
        # Apply color to the object in the mask
        color_mask[MOTS_mask == obj_id] = np.random.randint(0, 255, size=3)

    # Blend the color mask with the original image
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, color_mask, alpha, 0)

    # Create a copy to add text labels
    labeled_image = overlay.copy()

    # Define class names for class_id
    class_names = {1: 'Car', 2: 'Pedestrian'}

    # Annotate the image with class_id and object_instance_id
    for obj_id in obj_ids:
        if obj_id == 0:
            continue  # Skip background
        
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000

        if class_id not in [1, 2]:
            continue # Skip 10000?

        class_name = class_names.get(class_id, 'Unknown')

        # Find the position to place the label (centroid of the object mask)
        mask = (MOTS_mask == obj_id)
        y, x = np.where(mask)  # Get object coordinates
        if len(x) > 0 and len(y) > 0:
            centroid_x = int(np.mean(x))
            centroid_y = int(np.mean(y))

            # Create the label text
            label = f"{class_name} (ID: {obj_instance_id})"
            
            # Put the text on the image
            cv2.putText(labeled_image, label, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled_image

for i in range(446):
    if 0 <= i <= 9:
        rgb_path = 'MOTS_sequence_demo/0001_rgb/00000' + str(i) + '.png'
        gt_path = 'MOTS_sequence_demo/0001_MOTS_mask/00000' + str(i) + '.png'
        labeled_image = Combine_MOTS_image(rgb_path, gt_path)
        cv2.imwrite('MOTS_sequence_demo/0001_MOTS_combined/00000' + str(i) + '.png', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    elif 10 <= i <= 99:
        rgb_path = 'MOTS_sequence_demo/0001_rgb/0000' + str(i) + '.png'
        gt_path = 'MOTS_sequence_demo/0001_MOTS_mask/0000' + str(i) + '.png'
        labeled_image = Combine_MOTS_image(rgb_path, gt_path)
        cv2.imwrite('MOTS_sequence_demo/0001_MOTS_combined/0000' + str(i) + '.png', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))

    else:
        rgb_path = 'MOTS_sequence_demo/0001_rgb/000' + str(i) + '.png'
        gt_path = 'MOTS_sequence_demo/0001_MOTS_mask/000' + str(i) + '.png'
        labeled_image = Combine_MOTS_image(rgb_path, gt_path)
        cv2.imwrite('MOTS_sequence_demo/0001_MOTS_combined/000' + str(i) + '.png', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
