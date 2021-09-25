from alexandria import detection, ocr

# user input
confidence = 0.6

# detect books
classes, colors = detection.load_classes("coco.names")
model = detection.load_model(cfg='yolov3.cfg', model='yolov3.weights')
images_paths, images_list = detection.load_images()
outputs_list = [detection.detect(i, model) for i in images_list]
boxes_positions = [detection.get_boxes(i, o,
                    confidence, classes, colors)
                    for i, o in zip(images_list, outputs_list)]
for p, i, b in zip(images_paths, images_list, boxes_positions):
    print(p)
    detection.show_img_rectangles(i, b)


for path, image, boxes in zip(images_paths, images_list, boxes_positions):
    print(path)
    # returns a iterator
    sub_images = ocr.get_boxes_per_image(image, boxes)
    for i, img in enumerate(sub_images):
        d = ocr.image_to_text(img)
    # add ocr call
