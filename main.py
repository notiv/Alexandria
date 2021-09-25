from matplotlib import pyplot as plt
from requests.api import post
from alexandria import detection, ocr, post_processing
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")
# user input
confidence = 0.5
api_key = "../api_key.txt"
api_key = post_processing.get_api_key(api_key)


# detect books
classes, colors = detection.load_classes("coco.names")
model = detection.load_model(cfg='yolov3.cfg', model='yolov3.weights')
images_paths, images_list = detection.load_images()
# images_paths, images_list = images_paths[:1], images_list[:1]
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
    out["cleaned_titles"] = post_processing.clean_up_text(out["title_from_ocr"])
    out["final_titles"] = post_processing.search_book(out["cleaned_titles"], api_key)
    return out

books_text = {}
for path, image, boxes in tqdm(zip(images_paths, images_list, boxes_positions), total=len(images_paths)):
    books_text[path] = []
    sub_images = ocr.get_boxes_per_image(
        image=ocr.preprocess4ocr(image),
        boxes=boxes)

    for n, i in enumerate(sub_images):
        try:
            books_text[path].append(get_titles(n, i))
        except Exception:
            warnings.warn(f"{path}: {n} box is discarded")


for k, v in books_text.items():
    print(k)
    for i in v:
        print("orig", i["title_from_ocr"])
        print("clean", i["cleaned_titles"])
        print("final", i["final_titles"])

# # plt.imshow(ocr.preprocess4ocr(image))
# # image.shape
# # ocr.preprocess4ocr(image).shape


# # from autocorrect import Speller

# # spell = Speller(only_replacements=True)
# # w = spell("fournier tpeilly deep learning coders with fastaij")
# # post_processing.search_book(w, api_key)

# # [spell(i) for i in ["fournier", "tpeilly", "deep", "learning", "coders", "with", "fastaij"]]

# # print("orig", i["title_from_ocr"])
# # print("clean", i["cleaned_titles"])
# # print("final", i["final_titles"])

# # from textblob import TextBlob

# # ww = TextBlob("inas datubase internals petrov oreilly")

# # from alexandria import detection, ocr, post_processing
# # api_key = "../api_key.txt"
# # api_key = post_processing.get_api_key(api_key)
# # post_processing.search_book("inns database internal petrov oreille", api_key)


# # inns database internal petrov oreille


# # import textcleaner as tc

# # help(tc.document)

