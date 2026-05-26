def dice_score(image1, image2, mask):
    image1_list = image_to_list(image1)
    image2_list = image_to_list(image2)
    mask_list = image_to_list(mask)
    image_size = len(image1_list)
    mask_zeros = 0
    intersection_size = 0
    for i in range(image_size):
        if mask_list[i] == 1:
            if image1_list[i] == image2_list[i]:
                intersection_size += 1
        else:
            mask_zeros += 1
    return intersection_size/(image_size - mask_zeros)

def image_to_list(image):
    return [x for xs in image for x in xs]