import os
import time
import boto3
import pyhtml2md
import argparse
import requests
import nest_asyncio
import unstructured_client

from tqdm import tqdm

from llama_parse import LlamaParse
from llama_index.core import Settings
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.llms.anthropic import Anthropic
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal

from unstructured_client.models import operations, shared

from utils import get_image_path_list, save_txt

LLAMAPARSE_API_KEY = ""
OPENAI_API_KEY = ""
UPSTAGE_API_KEY = ""
UNSTRUCTURED_API_KEY = ""

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""


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


def infer_unstructured(image_dir, save_dir):
    start_time = time.time()

    client = unstructured_client.UnstructuredClient(
        api_key_auth=UNSTRUCTURED_API_KEY,
        server_url="https://api.unstructuredapp.io/general/v0/general",
    )
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):
        with open(image_path, "rb") as f:
            data = f.read()

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=data,
                    file_name=str(image_path),
                ),
                # --- Other partition parameters ---
                strategy=shared.Strategy.HI_RES,
                pdf_infer_table_structure=True,
                coordinates=True,
                languages=["eng", "kor"],
            ),
        )

        res = client.general.partition(request=req)
        elements = res.elements
        texts = ""
        for elem in elements:
            text = elem["text"]

            texts += text + "\n"

        save_txt(image_path, image_dir, save_dir, texts)

    total_time = time.time() - start_time
    print("time: ", total_time)
    print("avg time:: ", total_time / len(image_path_list))

    txt_save_path = os.path.join(save_dir, "time.txt")
    with open(txt_save_path, "w") as f:
        f.write("time: {}\n".format(total_time))
        f.write("avg time: {}\n".format(total_time / len(image_path_list)))


def infer_aws(image_dir, save_dir):
    def start_job(object_name):
        filename_with_ext = os.path.basename(object_name)

        s3.Bucket(s3_bucket_name).upload_file(object_name, filename_with_ext)

        response = None
        response = client.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket_name,
                    "Name": filename_with_ext
                }
            },
            FeatureTypes = ["LAYOUT", "TABLES"]
        )

        return response["JobId"]

    def is_job_complete(job_id):
        time.sleep(1)
        response = client.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

        while(status == "IN_PROGRESS"):
            time.sleep(1)
            response = client.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]
            print("Job status: {}".format(status))

        return status

    def get_job_results(job_id):
        pages = []
        time.sleep(1)
        response = client.get_document_analysis(JobId=job_id)
        pages.append(response)
        print("Resultset page received: {}".format(len(pages)))
        next_token = None
        if "NextToken" in response:
            next_token = response["NextToken"]

        while next_token:
            time.sleep(1)
            response = client.\
                get_document_analysis(JobId=job_id, NextToken=next_token)
            pages.append(response)
            print("Resultset page received: {}".format(len(pages)))
            next_token = None
            if "NextToken" in response:
                next_token = response["NextToken"]

        return pages

    def post_process(data):
        def get_text(result, blocks_map):
            text = ""
            if "Relationships" in result:
                for relationship in result["Relationships"]:
                    if relationship["Type"] == "CHILD":
                        for child_id in relationship["Ids"]:
                            word = blocks_map[child_id]
                            if word["BlockType"] == "WORD":
                                text += " " + word["Text"]
            return text[1:]

        all_elems = {}
        for elem in data["Blocks"]:
            _id = elem["Id"]
            all_elems[_id] = elem

        processed_list = []
        for idx, elem in enumerate(data["Blocks"]):
            if elem["BlockType"] == "LAYOUT_LIST":
                continue

            if "LAYOUT" in elem["BlockType"] and elem["BlockType"] != "LAYOUT_TABLE":

                bbox = elem["Geometry"]["BoundingBox"]

                x = bbox["Left"]
                y = bbox["Top"]
                w = bbox["Width"]
                h = bbox["Height"]

                coord = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                xy_coord = [{"x": x, "y": y} for x, y in coord]

                # category = CATEGORY_MAP.get(elem["BlockType"], "paragraph")
                category = elem["BlockType"]

                transcription = ""

                if elem["BlockType"] != "LAYOUT_FIGURE":
                    for item in all_elems[elem["Id"]]["Relationships"]:
                        for id_ in item["Ids"]:
                            if all_elems[id_]["BlockType"] == "LINE":
                                word = all_elems[id_]["Text"]
                                transcription += word + "\n"

                data_dict = {
                    "coordinates": xy_coord,
                    "category": category,
                    "id": idx,
                    "content": {
                        "text": transcription,
                        "html": "",
                        "markdown": ""
                    }
                }
                processed_list.append(data_dict)

            elif elem["BlockType"] == "TABLE":

                bbox = elem["Geometry"]["BoundingBox"]

                x = bbox["Left"]
                y = bbox["Top"]
                w = bbox["Width"]
                h = bbox["Height"]

                coord = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                xy_coord = [{"x": x, "y": y} for x, y in coord]

                # category = CATEGORY_MAP.get(elem["BlockType"], "paragraph")
                category = elem["BlockType"]

                table_cells = {}
                for relationship in elem["Relationships"]:
                    if relationship["Type"] == "CHILD":
                        for cell_id in relationship["Ids"]:
                            cell_block = next((block for block in data["Blocks"] if block["Id"] == cell_id), None)
                            if cell_block is not None and cell_block["BlockType"] == "CELL":
                                row_index = cell_block["RowIndex"] - 1
                                column_index = cell_block["ColumnIndex"] - 1
                                row_span = cell_block["RowSpan"]
                                column_span = cell_block["ColumnSpan"]
                                table_cells[(row_index, column_index)] = {
                                    "block": cell_block,
                                    "span": (row_span, column_span),
                                    "text": get_text(cell_block, all_elems),
                                }
                max_row_index = max(cell[0] for cell in table_cells.keys())
                max_column_index = max(cell[1] for cell in table_cells.keys())
                for relationship in elem["Relationships"]:
                    if relationship["Type"] == "MERGED_CELL":
                        for cell_id in relationship["Ids"]:
                            cell_block = next((block for block in data["Blocks"] if block["Id"] == cell_id), None)
                            if cell_block is not None and cell_block["BlockType"] == "MERGED_CELL":
                                row_index = cell_block["RowIndex"] - 1
                                column_index = cell_block["ColumnIndex"] - 1
                                row_span = cell_block["RowSpan"]
                                column_span = cell_block["ColumnSpan"]
                                for i in range(row_span):
                                    for j in range(column_span):
                                        del table_cells[(row_index + i, column_index + j)]
                                text = ""
                                for child_ids in cell_block["Relationships"][0]["Ids"]:
                                    child_cell_block = next((block for block in data["Blocks"] if block["Id"] == child_ids), None)
                                    text += " " + get_text(child_cell_block, all_elems)
                                table_cells[(row_index, column_index)] = {
                                    "block": cell_block,
                                    "span": (row_span, column_span),
                                    "text": text[1:],
                                }
                html_table = "<table>"

                for row_index in range(max_row_index + 1):
                    html_table += "<tr>"
                    for column_index in range(max_column_index + 1):
                        cell_data = table_cells.get((row_index, column_index))
                        if cell_data:
                            cell_block = cell_data["block"]
                            row_span, column_span = cell_data["span"]

                            cell_text = cell_data["text"]
                            html_table += f"<td rowspan='{row_span}' colspan='{column_span}''>{cell_text}</td>"
                    html_table += "</tr>"
                html_table += "</table>"

                data_dict = {
                    "coordinates": xy_coord,
                    "category": category,
                    "id": idx,
                    "content": {
                        "text": "",
                        "html": html_table,
                        "markdown": ""
                    }
                }
                processed_list.append(data_dict)

        return processed_list

    s3 = boto3.resource("s3")

    client = boto3.client(
        "textract",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    start_time = time.time()
    image_path_list = get_image_path_list(image_dir)
    for image_path in tqdm(image_path_list):

        # for PDF input file
        # job_id = start_job(filepath)
        # print("Started job with id: {}".format(job_id))
        # if is_job_complete(job_id):
        #     result = get_job_results(job_id)

        with open(image_path, "rb") as file:
            img_test = file.read()
            bytes_test = bytearray(img_test)

        result = client.analyze_document(
            Document={"Bytes": bytes_test},
            FeatureTypes = ["LAYOUT", "TABLES"]
        )

        processed_list = post_process(result)

        texts = ""
        for elem in processed_list:
            if elem["content"]["html"]:
                text = pyhtml2md.convert(elem["content"]["html"]) + "\n"
            else:
                text = elem["content"]["text"] + "\n"

            texts += text

        save_txt(image_path, image_dir, save_dir, texts)

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
    elif args.model == "unstructured":
        infer_unstructured(args.image_dir, args.save_dir)
    elif args.model == "aws":
        infer_aws(args.image_dir, args.save_dir)
    else:
        raise ValueError("Invalid model name")
