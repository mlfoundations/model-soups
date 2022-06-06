import json
from pathlib import Path


if __name__ == '__main__':
    METADATA = Path(__file__).parent / 'objectnet_metadata'

    with open(METADATA / 'folder_to_objectnet_label.json', 'r') as f:
        folder_map = json.load(f)
        folder_map = {v: k for k, v in folder_map.items()}

    with open(METADATA / 'objectnet_to_imagenet_1k.json', 'r') as f:
        objectnet_map = json.load(f)

    with open(METADATA / 'imagenet_to_labels.json', 'r') as f:
        imagenet_map = json.load(f)
        imagenet_map = {v: k for k, v in imagenet_map.items()}

    folder_to_ids, class_sublist = {}, []
    for objectnet_name, imagenet_names in objectnet_map.items():
        imagenet_names = imagenet_names.split('; ')
        imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
        class_sublist.extend(imagenet_ids)
        folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

    class_sublist = sorted(class_sublist)
    class_sublist_mask = [(i in class_sublist) for i in range(1000)]
    folder_map = {k: [class_sublist.index(x) for x in v] for k, v in folder_to_ids.items()}

    print('here')