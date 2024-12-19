import os
import time
import argparse
import requests
import nest_asyncio

from tqdm import tqdm

from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.llms.anthropic import Anthropic
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal

from utils import get_image_path_list, save_txt

LLAMAPARSE_API_KEY = ""
OPENAI_API_KEY = ""
UPSTAGE_API_KEY = ""


def infer_llamaparse(image_dir, save_dir):
    parser = LlamaParse(
        api_key=LLAMAPARSE_API_KEY,
        result_type="markdown",
        premium_mode=True  # "markdown" and "text" are available
    )
    start_time = time.time()
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):
        try:
            documents = parser.load_data(str(image_path))[0].text
            save_txt(image_path, image_dir, save_dir, documents)
        except IndexError:
            print("Error processing file {}...".format(image_path))
            pass
    total_time = time.time() - start_time
    print("time: ", total_time)
    print("avg time:: ", total_time / len(image_path_list))

    txt_save_path = os.path.join(save_dir, "time.txt")
    with open(txt_save_path, "w") as f:
        f.write("time: {}\n".format(total_time))
        f.write("avg time: {}\n".format(total_time / len(image_path_list)))


def infer_sonnet(image_dir, save_dir):
    parser = LlamaParse(
        api_key=LLAMAPARSE_API_KEY,
        result_type="markdown",  # "markdown" and "text" are available
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="anthropic-sonnet-3.5",
    )

    start_time = time.time()
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):
        try:
            documents = parser.load_data(str(image_path))[0].text
            save_txt(image_path, image_dir, save_dir, documents)
        except IndexError:
            pass
    total_time = time.time() - start_time
    print("time: ", total_time)
    print("avg time:: ", total_time / len(image_path_list))

    txt_save_path = os.path.join(save_dir, "time.txt")
    with open(txt_save_path, "w") as f:
        f.write("time: {}\n".format(total_time))
        f.write("avg time: {}\n".format(total_time / len(image_path_list)))


def infer_gpt4o(image_dir, save_dir):
    parser = LlamaParse(
        api_key=LLAMAPARSE_API_KEY,
        result_type="markdown",  # "markdown" and "text" are available
        use_vendor_multimodal_model=True,
        vendor_multimodal_model="openai-gpt4o",
        gpt4o_api_key=OPENAI_API_KEY,
    )

    start_time = time.time()
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):
        try:
            documents = parser.load_data(str(image_path))[0].text
            save_txt(image_path, image_dir, save_dir, documents)
        except IndexError:
            pass
    total_time = time.time() - start_time
    print("time: ", total_time)
    print("avg time:: ", total_time / len(image_path_list))

    txt_save_path = os.path.join(save_dir, "time.txt")
    with open(txt_save_path, "w") as f:
        f.write("time: {}\n".format(total_time))
        f.write("avg time: {}\n".format(total_time / len(image_path_list)))


def infer_upstage(image_dir, save_dir):
    endpoint = "https://api.upstage.ai/v1/document-ai/document-parse"
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
    }
    output_formats = ["text", "html", "markdown"]
    data = {
        "ocr": "force",
        "model": "document-parse-240910",
        "output_formats": f"{output_formats}"
    }

    start_time = time.time()
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):

        files = {
            "document": open(image_path, "rb"),
        }

        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
            data=data
        )
        json_result = response.json()
        save_txt(image_path, image_dir, save_dir, json_result["content"]["markdown"])

    total_time = time.time() - start_time
    print("time: ", total_time)
    print("avg time:: ", total_time / len(image_path_list))

    txt_save_path = os.path.join(save_dir, "time.txt")
    with open(txt_save_path, "w") as f:
        f.write("time: {}\n".format(total_time))
        f.write("avg time: {}\n".format(total_time / len(image_path_list)))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--image_dir",
        type=str,
        default="Path to the image directory.",
    )
    args.add_argument(
        "--model",
        type=str,
        default="llamaparse",
        help="Model name to use."
    )
    args.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save the results."
    )
    args = args.parse_args()

    start_t = time.perf_counter()
    if args.model == "llamaparse":
        infer_llamaparse(args.image_dir, args.save_dir)
    elif args.model == "sonnet":
        infer_sonnet(args.image_dir, args.save_dir)
    elif args.model == "gpt4o":
        infer_gpt4o(args.image_dir, args.save_dir)
    elif args.model == "upstage":
        infer_upstage(args.image_dir, args.save_dir)
    else:
        raise ValueError("Invalid model name")
