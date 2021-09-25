import pytesseract

from pytesseract import Output

def image_to_text(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    text = d.get('text')

    return text

def get_boxes_per_image(image, boxes):
    return (image[slice(*b.y_slice), slice(*b.x_slice)] for b in boxes)
