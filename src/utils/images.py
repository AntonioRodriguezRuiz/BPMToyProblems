from PIL import Image, ImageFile
from io import BytesIO
import base64

MAX_HEIGHT = 1920
MAX_WIDTH = 1080


# Convert Image to Base64
def im_2_b64(image: Image.Image) -> str:
    buff = BytesIO()
    aspect_ratio = image.size[0] / image.size[1]
    if image.size[0] > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, int(MAX_WIDTH / aspect_ratio)))
    if image.size[1] > MAX_HEIGHT:
        image = image.resize((int(MAX_HEIGHT * aspect_ratio), MAX_HEIGHT))
    image.convert("RGB").save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str


def resize_img(image, new_width=MAX_WIDTH, new_height=MAX_HEIGHT) -> Image.Image:
    aspect_ratio = image.size[0] / image.size[1]
    if image.size[0] > new_width:
        image = image.resize((new_width, int(new_width / aspect_ratio)))
    if image.size[1] > new_height:
        image = image.resize((int(new_height * aspect_ratio), new_height))
    return image


# Convert Base64 to Image
def b64_2_img(data) -> ImageFile:
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def add_padding(image, horizontal_padding, vertical_padding):
    """
    Add black padding to an image with specified horizontal and vertical padding amounts.
    Places the original image in the center of the new padded image.

    Args:
        image: PIL Image to pad
        horizontal_padding: Padding to add to left and right (total horizontal padding will be 2x this value)
        vertical_padding: Padding to add to top and bottom (total vertical padding will be 2x this value)

    Returns:
        Padded PIL Image
    """
    width, height = image.size
    new_width = width + (horizontal_padding * 2)
    new_height = height + (vertical_padding * 2)

    # Create a new black image with the padded dimensions
    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Paste the original image in the center
    padded_image.paste(image, (horizontal_padding, vertical_padding))

    return padded_image
