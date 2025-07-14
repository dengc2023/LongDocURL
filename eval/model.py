import torch
import base64
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from abc import ABC, abstractmethod
from openai import OpenAI
import requests
import os
from typing import Union
import oss2
import json

# TODO
project_prefix = "/mnt/workspace/Projects/CodeLib/LongDocURL/"
config_file = os.path.join(project_prefix, "config/api_config.json")

class APIInferencer(ABC):
    def __init__(self):
        pass
        # uncomment if oss paths are used
        # self.bucket = self.get_alimama_oss_bucket()

    def get_alimama_oss_bucket(self):
        # TODO
        endpoint = ''
        access_key_id = ''
        access_key_secret = ''
        bucket_name = ''
        bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
        return bucket

    @abstractmethod
    def infer(self, prompt: str, image_path: str) -> str:
        pass

    def load_client(self):
        with open(config_file, "r", encoding="utf-8") as rf:
            config = json.load(rf)
        return OpenAI(api_key=config["gpt4o"]["access_key"], base_url=config["gpt4o"]["base_url"])

    def cleanup(self):
        if hasattr(self, 'client'):
            del self.client

    def encode_image_to_base64(self, image_path: str) -> str:
        if 'https' in image_path:
            response = requests.get(image_path)
            img = BytesIO(response.content)
            return base64.b64encode(img.read()).decode('utf-8')

        if image_path.startswith('oss://'):
            return base64.b64encode(self.bucket.get_object(image_path[6:].split("/", 1)[1]).read()).decode("utf-8")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_correct_response(self, model_name: str, prompt: str, image_path: Union[list, str]) -> str:
        response = self.model_chat(model_name, prompt, image_path)
        return response

    def model_chat(self, model_name: str, prompt: str, image_path: str) -> str:
        client = self.load_client()
        messages = [
            {
                "role": "user",
                "content": self.build_message_content(prompt, image_path)
            }
        ]
        max_try = 2
        response = None
        while response is None and max_try > 0:
            try:
                completion = client.chat.completions.create(model=model_name, messages=messages, temperature=0.)
                response = completion.choices[0].message.content
            except Exception as e:
                print("exception: ", e)
                max_try -= 1
        return response

    def build_message_content(self, prompt: str, image_path: str):
        content = [{"type": "text", "text": prompt}]
        if image_path is None:
            return content
        if isinstance(image_path, str):
            image_paths = [image_path]
        elif isinstance(image_path, Union[list, tuple]):
            image_paths = image_path
        base64_images = [self.encode_image_to_base64(image_path) for image_path in image_paths]
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
        return content


class OpenLVLMsInferencer(ABC):
    def __init__(self):
        pass
        # uncomment if oss paths are used
        # self.bucket = self.get_alimama_oss_bucket()

    @abstractmethod
    def infer(self, prompt: str, image_path: str) -> str:
        pass

    def load_client(self):
        with open(config_file, "r", encoding="utf-8") as rf:
            config = json.load(rf)
        return OpenAI(api_key=config["gpt4o"]["access_key"], base_url=config["gpt4o"]["base_url"])

    def cleanup(self):
        if hasattr(self, 'client'):
            del self.client

    def encode_image_to_base64(self, image_path: str) -> str:
        if 'https' in image_path:
            response = requests.get(image_path)
            img = BytesIO(response.content)
            return base64.b64encode(img.read()).decode('utf-8')

        if image_path.startswith('oss://'):
            return base64.b64encode(self.bucket.get_object(image_path[6:].split("/", 1)[1]).read()).decode("utf-8")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_correct_response(self, model_name: str, prompt: str, image_path: Union[list, str], **kwargs) -> str:
        response = self.model_chat(model_name, prompt, image_path, **kwargs)
        return response

    def model_chat(self, model_name: str, prompt: str, image_path: str, **kwargs) -> str:
        device = kwargs.pop("device", "cpu")
        ckpt_path = kwargs.pop("local_path", None)
        model_pool = kwargs.pop("model_pool", {})

        messages = [
            {
                "role": "user",
                "content": self.build_message_content(prompt=prompt, image_path=image_path),
            }
        ]

        if model_name in ["qwen2-vl-7b", "qwen25-vl-7b"]:
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     ckpt_path, torch_dtype="auto", device_map=device
            # )

            # model_key = f"{model_name}_{device}"  # combine model name and device ID as unique key
            model_key = f"{model_name}"  # use model name as unique key
            if model_key not in model_pool:
                if model_name in ["qwen2-vl-7b"]:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        ckpt_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="cpu",
                    )
                elif model_name in ["qwen25-vl-7b"]:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        ckpt_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="cpu",
                    )
                else:
                    raise ValueError("The model name specified does not exists!")
                processor = AutoProcessor.from_pretrained(ckpt_path)
                model_pool[model_key] = (model, processor)
            else:
                model, processor = model_pool[model_key]
            model = model.to(device)
   
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return output_text
        else:
            raise ValueError("The model specified does not exists!")

    def build_message_content(self, prompt: str, image_path: str):
        content = [{"type": "text", "text": prompt}]
        if image_path is None:
            return content
        if isinstance(image_path, str):
            image_paths = [image_path]
        elif isinstance(image_path, Union[list, tuple]):
            image_paths = image_path
        base64_images = [self.encode_image_to_base64(image_path) for image_path in image_paths]
        for i, base64_image in enumerate(base64_images):
            content += [
                {"type": "text", "text": f"Below is the {i+1}-th image (total {len(base64_images)} images).\n"},
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{base64_image}",
                },
            ]
        return content


class QwenMaxInferencer(APIInferencer):
    def infer(self, prompt: str, image_path: str) -> str:
        response = self.get_correct_response('qwen_max', prompt, image_path)
        return response


class O1PreviewInferencer(APIInferencer):
    def infer(self, prompt: str, image_path: str) -> str:
        response = self.get_correct_response('o1-preview-0912', prompt, image_path)
        return response


class GPT4oInferencer(APIInferencer):
    def infer(self, prompt: str, image_path: str) -> str:
        response = self.get_correct_response('gpt-4o-0513', prompt, image_path)
        # response = self.get_correct_response('gpt-4o', prompt, image_path)
        return response


class Gemini15ProInferencer(APIInferencer):
    def infer(self, prompt: str, image_path: str) -> str:
        response = self.get_correct_response('gemini-1.5-pro', prompt, image_path)
        return response


class QwenVLMaxInferencer(APIInferencer):
    def infer(self, prompt: str, image_path: str) -> str:
        response = self.get_correct_response('qwen-vl-max', prompt, image_path)
        return response
    

class Qwen2VLInferencer(OpenLVLMsInferencer):
    def __init__(self, model_name):
        self.model_name = model_name
        self.ckpt_paths = {
            "qwen2-vl-7b": "/data1/model/Qwen/Qwen2-VL-7B-Instruct",
            "qwen25-vl-7b": "/data1/model/Qwen/Qwen2.5-VL-7B-Instruct"
        }
    def infer(self, prompt: str, image_path: str, **kwargs) -> str:
        if self.model_name in self.ckpt_paths:
            kwargs.update({"local_path": self.ckpt_paths[self.model_name]})
        response = self.get_correct_response(self.model_name, prompt, image_path, **kwargs)
        return response
