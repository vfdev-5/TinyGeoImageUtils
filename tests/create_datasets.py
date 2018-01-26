
import os


def create_dataset_with_target_is_folder(tempdir, add_garbage_files=False, n=10):
    """
    Create a test dataset with images and masks
    root
        - cls1
            - id_0.ext1
            - id_1.ext1
            ...
            (optional)
            - garbage_id_0.ext3
            - garbage_id_1.ext3
            ...
        - cls2
            - id_0.ext1
            - id_1.ext1
            ...
            (optional)
            - garbage_id_0.ext3
            - garbage_id_1.ext3
            ...
        - cls3
            - id_0.ext1
            - id_1.ext1
            ...
            (optional)
            - garbage_id_0.ext3
            - garbage_id_1.ext3
            ...
    """
    dataset = []

    def _create_data(path, cls_id, add_garbage_files):
        os.mkdir(path)
        for i in range(n):
            file_path = os.path.join(path, "id_%i.ext1" % i)
            with open(file_path, 'w') as f:
                f.write('test')
            dataset.append((file_path, cls_id))
        if add_garbage_files:
            for i in range(n//2):
                garbage_filepath = os.path.join(path, "garbage_id_%i.ext3" % i)
                with open(garbage_filepath, 'w') as f:
                    f.write('test')

    cls1_path = os.path.join(tempdir, "cls_1")
    _create_data(cls1_path, "cls_1", add_garbage_files=add_garbage_files)
    cls2_path = os.path.join(tempdir, "cls_2")
    _create_data(cls2_path, "cls_2", add_garbage_files=add_garbage_files)
    cls3_path = os.path.join(tempdir, "cls_3")
    _create_data(cls3_path, "cls_3", add_garbage_files=add_garbage_files)

    return dataset


def create_dataset_with_target_is_mask_file(tempdir, add_garbage_files=False, img_mask_same_ext=False, n=30):
    """
    Create a test dataset with images and masks
    root
        - images_and_masks
            - id_0.img_ext
            - label_id_0.mask_ext
            - id_1.img_ext
            - label_id_1.mask_ext
            ...
            (optional)
            - garbage_id_0.ext3
            - garbage_id_1.ext3
            ...
    """
    dataset = []

    path = os.path.join(tempdir, "images_and_masks")
    os.mkdir(path)

    img_ext = "ext1"
    mask_ext = "ext2" if not img_mask_same_ext else img_ext

    for i in range(n):
        img_filepath = os.path.join(path, "id_%i.%s" % (i, img_ext))
        mask_filepath = os.path.join(path, "label_id_%i.%s" % (i, mask_ext))
        with open(img_filepath, 'w') as f:
            f.write('test')
        with open(mask_filepath, 'w') as f:
            f.write('test')
        dataset.append((img_filepath, mask_filepath))

    if add_garbage_files:
        for i in range(n//2):
            garbage_filepath = os.path.join(path, "garbage_id_%i.ext3" % i)
            with open(garbage_filepath, 'w') as f:
                f.write('test')

    return dataset


def create_dataset_with_target_is_mask_file2(tempdir, add_garbage_files=False, img_mask_same_ext=False, n=30):
    """
    Create a test dataset with images and masks
    root
        - images
            - id_0.img_ext
            - id_1.img_ext
            ...
            (optional)
            - garbage_id_0.ext3
            - garbage_id_1.ext3
            ...
        - masks
            - id_0.mask_ext
            - id_1.mask_ext
            ...
            (optional)
            - garbage_id_0.ext4
            - garbage_id_1.ext4
            ...
    """
    dataset = []
    img_path = os.path.join(tempdir, "images")
    os.mkdir(img_path)

    mask_path = os.path.join(tempdir, "masks")
    os.mkdir(mask_path)

    img_ext = "ext1"
    mask_ext = "ext2" if not img_mask_same_ext else img_ext
    for i in range(n):
        img_filepath = os.path.join(img_path, "id_%i.%s" % (i, img_ext))
        mask_filepath = os.path.join(mask_path, "id_%i.%s" % (i, mask_ext))
        with open(img_filepath, 'w') as f:
            f.write('test')
        with open(mask_filepath, 'w') as f:
            f.write('test')
        dataset.append((img_filepath, mask_filepath))

    if add_garbage_files:
        for i in range(n//2):
            garbage_filepath = os.path.join(img_path, "garbage_id_%i.ext3" % i)
            with open(garbage_filepath, 'w') as f:
                f.write('test')
        for i in range(n//2):
            garbage_filepath = os.path.join(mask_path, "garbage_id_%i.ext4" % i)
            with open(garbage_filepath, 'w') as f:
                f.write('test')

    return dataset


def create_potsdam_like_dataset(tempdir):
    """
    Create a dataset like potsdam with images and masks
    root
        - images
            - top_potsdam_2_10_RGB.tif
            - top_potsdam_4_13_RGB.tif
            - top_potsdam_6_14_RGB.tif
            ...
        - labels
            - top_potsdam_2_10_label.tif
            - top_potsdam_6_10_label.tif
            - top_potsdam_2_11_label.tif
            ...
    """
    dataset = []
    img_path = os.path.join(tempdir, "images")
    os.mkdir(img_path)

    mask_path = os.path.join(tempdir, "labels")
    os.mkdir(mask_path)

    image_files = ["top_potsdam_2_10_RGB.tif",  "top_potsdam_4_13_RGB.tif",  "top_potsdam_6_14_RGB.tif",
        "top_potsdam_2_11_RGB.tif",  "top_potsdam_4_14_RGB.tif",  "top_potsdam_6_15_RGB.tif",
        "top_potsdam_2_12_RGB.tif",  "top_potsdam_4_15_RGB.tif",  "top_potsdam_6_7_RGB.tif",
        "top_potsdam_2_13_RGB.tif",  "top_potsdam_5_10_RGB.tif",  "top_potsdam_6_8_RGB.tif",
        "top_potsdam_2_14_RGB.tif",  "top_potsdam_5_11_RGB.tif",  "top_potsdam_6_9_RGB.tif",
        "top_potsdam_3_10_RGB.tif",  "top_potsdam_5_12_RGB.tif",  "top_potsdam_7_10_RGB.tif",
        "top_potsdam_3_11_RGB.tif",  "top_potsdam_5_13_RGB.tif",  "top_potsdam_7_11_RGB.tif",
        "top_potsdam_3_12_RGB.tif",  "top_potsdam_5_14_RGB.tif",  "top_potsdam_7_12_RGB.tif",
        "top_potsdam_3_13_RGB.tif",  "top_potsdam_5_15_RGB.tif",  "top_potsdam_7_13_RGB.tif",
        "top_potsdam_3_14_RGB.tif",  "top_potsdam_6_10_RGB.tif",  "top_potsdam_7_7_RGB.tif",
        "top_potsdam_4_10_RGB.tif",  "top_potsdam_6_11_RGB.tif",  "top_potsdam_7_8_RGB.tif",
        "top_potsdam_4_11_RGB.tif",  "top_potsdam_6_12_RGB.tif",  "top_potsdam_7_9_RGB.tif",
        "top_potsdam_4_12_RGB.tif",  "top_potsdam_6_13_RGB.tif"
    ]

    mask_files = [
        "top_potsdam_2_10_label.tif",  "top_potsdam_6_10_label.tif",
        "top_potsdam_2_11_label.tif",  "top_potsdam_6_11_label.tif",
        "top_potsdam_2_12_label.tif",  "top_potsdam_6_12_label.tif",
        "top_potsdam_3_10_label.tif",  "top_potsdam_6_7_label.tif",
        "top_potsdam_3_11_label.tif",  "top_potsdam_6_8_label.tif",
        "top_potsdam_3_12_label.tif",  "top_potsdam_6_9_label.tif",
        "top_potsdam_4_10_label.tif",  "top_potsdam_7_10_label.tif",
        "top_potsdam_4_11_label.tif",  "top_potsdam_7_11_label.tif",
        "top_potsdam_4_12_label.tif",  "top_potsdam_7_12_label.tif",
        "top_potsdam_5_10_label.tif",  "top_potsdam_7_7_label.tif",
        "top_potsdam_5_11_label.tif",  "top_potsdam_7_8_label.tif",
        "top_potsdam_5_12_label.tif",  "top_potsdam_7_9_label.tif",
    ]

    for f in mask_files:
        mask_filepath = os.path.join(mask_path, f)
        with open(mask_filepath, 'w') as handler:
            handler.write('test')
        f = f[:-9] + "RGB.tif"
        img_filepath = os.path.join(img_path, f)
        with open(img_filepath, 'w') as handler:
            handler.write('test')
        image_files.remove(f)
        dataset.append((img_filepath, mask_filepath))

    return dataset
