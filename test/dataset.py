import os

import numpy as np

def save_npz(filename: str, **kwargs):
    with open(filename, 'wb') as f:
        print("Saving to", filename)
        np.savez(f, **kwargs)


def load_npz(filename: str):
    import numpy as np

    return np.load(filename)


if __name__ == "__main__":
    car3d_dir = "/NFS/database_personal/anomaly_detection/data/Cars3D"
    test_data_p = os.path.join(car3d_dir, "cars3d__az_id__test.npz")
    test_data_p = os.path.join(car3d_dir, "cars3d__az_id__train.npz")

    test_data = load_npz(test_data_p)
    imgs = test_data["imgs"]
    contents = test_data["contents"]
    classes = test_data["classes"]
    n_classes = test_data["n_classes"]
    anom_label = test_data["anom_label"]
    from PIL import Image
    image_dir = "images" + "/94"
    os.makedirs(image_dir, exist_ok=True)
    _imgs = imgs[contents==94]
    for i in range(20):
        Image.fromarray(_imgs[i], mode="RGB").save(
            os.path.join(image_dir, f"test_{i}.png")
        )
