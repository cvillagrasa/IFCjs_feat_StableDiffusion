import io
import base64
import numpy as np
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_diffusion_main import infer

img_size_pixels = 512

app = Flask(__name__)
cors = CORS(app)


def add_margins_to_img(pil_img, top, bottom, left, right, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def make_img_square(pil_img, color=(255, 255, 255)):
    width, height = pil_img.size
    margin = int((height - width) / 2)
    extra_pixel = 2 * margin + width - height
    top, bottom, left, right = [0 for _ in range(4)]
    if margin >= 0:
        left = margin
        right = margin + extra_pixel
    else:
        top = -margin
        bottom = -margin - extra_pixel
    return add_margins_to_img(pil_img, top, bottom, left, right, color)


def base64_to_image(img_str):
    data_b64 = img_str.split('base64,')[1]
    data_decoded = base64.b64decode(data_b64)
    return Image.open(io.BytesIO(data_decoded))


def image_to_base64(image):
    buff = io.BytesIO()
    image.save(buff, format='PNG')
    img_bytes = base64.b64encode(buff.getvalue())
    return 'data:image/png;base64,' + img_bytes.decode('utf8')


def image_when_busy(image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('roboto.ttf', 16)
    draw.text((140, 30), 'THE SERVER IS NOW BUSY', (255, 0, 0), font=font)
    draw.text((90, 90), 'PLEASE, TRY AGAIN IN SOME SECONDS', (255, 0, 0), font=font)
    draw.text((90, 150), 'I HAVE ONLY ONE GPU. CHEERS, Carlos', (255, 0, 0), font=font)
    return image


def prompt_to_text(prompt, label):
    with open(f'../res/{label}.txt', "w") as f:
        f.write(prompt)


@app.route('/receiver', methods=['POST'])
def postME():
    data = request.get_json()
    img = base64_to_image(data['image'])
    resize_factor = float(img_size_pixels) / np.max([img.size])
    if resize_factor < 1:
        img = img.resize(tuple([round(resize_factor * size) for size in img.size]))
    img = make_img_square(img)
    img_inpainted = infer(data['prompt'], img)
    if img_inpainted is None:
        img_inpainted = image_when_busy(img)
    img_inpainted_encoded = image_to_base64(img_inpainted)
    label = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_inpainted.save(f'../res/{label}.png')
    prompt_to_text(data['prompt'], label)
    print('image received!')
    return jsonify({'imgInpainted': img_inpainted_encoded})


if __name__ == "__main__":
    cert = '/home/carlos/Documentos/IFCjs/IFCjs_feat_StableDiffusion/ssl/fullchain.pem'
    key = '/home/carlos/Documentos/IFCjs/IFCjs_feat_StableDiffusion/ssl/privkey.pem'
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context=(cert, key))
