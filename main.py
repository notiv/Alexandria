from matplotlib import pyplot as plt
from requests.api import post
from alexandria import detection, ocr, post_processing
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("default")
# user input
confidence = 0.5
api_key = "../api_key.txt"
api_key = post_processing.get_api_key(api_key)


# detect books
classes, colors = detection.load_classes("coco.names")
model = detection.load_model(cfg='yolov3.cfg', model='yolov3.weights')
images_paths, images_list = detection.load_images()
outputs_list = [detection.detect(i, model) for i in images_list]
boxes_positions = [detection.get_boxes(i, o,
                    confidence, classes, colors)
                    for i, o in zip(images_list, outputs_list)]
# for p, i, b in zip(images_paths, images_list, boxes_positions):
#     print(p)
#     detection.show_img_rectangles(i, b)

def get_titles(id, img):
    out = {}
    out["id"] = id
    out["book_image"] = img
    out["title_from_ocr"] = " ".join(ocr.image_to_text(img))
    out["cleaned_titles"] = post_processing.search_book(out["title_from_ocr"], api_key)
    return out

books_text = {}
for path, image, boxes in tqdm(zip(images_paths, images_list, boxes_positions), total=len(images_paths)):
    books_text[path] = []
    sub_images = ocr.get_boxes_per_image(image, boxes)

    for n, i in enumerate(sub_images):
        try:
            books_text[path].append(get_titles(n, i))
        except ValueError:
            warnings.warn(f"{path}: {n} box is discarded as on the edge")


for k, v in books_text.items():
    print(k)
    for i in v:
        print(i["title_from_ocr"])
        print(i["cleaned_titles"])


