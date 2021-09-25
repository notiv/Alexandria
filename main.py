from alexandria import detection

# user input
confidence = 0.6

# detect books
classes, colors = detection.load_classes("coco.names")
model = detection.load_model(cfg='yolov3.cfg', model='yolov3.weights')
images_list = detection.load_images()
outputs_list = [detection.detect(i, model) for i in images_list]
boxes_positions = [detection.get_boxes(i, o,
                    confidence, classes, colors)
                    for i, o in zip(images_list, outputs_list)]
for i, b in zip(images_list, boxes_positions):
    detection.show_img_rectangles(i, b)
