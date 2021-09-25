import pytesseract

def get_boxes_per_image(image, boxes):
    return (image[slice(*b.y_slice), slice(*b.x_slice)] for b in boxes)
