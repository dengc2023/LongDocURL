import os
from io import BytesIO
import oss2
import base64
import json
import requests
import re
from typing import Union

# TODO
def get_alimama_oss_bucket():
    endpoint = ''
    access_key_id = ''
    access_key_secret = ''
    bucket_name = ''
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    return bucket

# # uncomment if oss paths are used
# bucket = get_alimama_oss_bucket()


def encode_image_to_base64(image_path):
    if 'https' in image_path:
        response = requests.get(image_path)
        img = BytesIO(response.content)
        return base64.b64encode(img.read()).decode('utf-8')

    if image_path.startswith('oss://'):
        return base64.b64encode(bucket.get_object(image_path[6:].split("/", 1)[1]).read()).decode("utf-8")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

        
def get_msg_format(prompt, img_urls):
    content = [{"type": "text", "text": prompt}]
    if img_urls is not None:
        if isinstance(img_urls, str):
            base64_images = [encode_image_to_base64(img_urls)]
        elif isinstance(img_urls, Union[list, tuple]):
            base64_images = [encode_image_to_base64(img_url) for img_url in img_urls]
        
        for i, base64_image in enumerate(base64_images):
            content += [
                {"type": "text", "text": f"Below is the {i+1}-th image (total {len(base64_images)} images).\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ]
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


