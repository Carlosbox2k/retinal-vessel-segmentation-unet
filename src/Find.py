import os

def find(name, is_file=False):
    for root, dirs, files in os.walk(os.getcwd()):
        if root.__contains__("retinal-vessel-segmentation-unet"):
            if (not is_file and name in dirs) or (is_file and name in files):
                return os.path.join(root, name)
    return None