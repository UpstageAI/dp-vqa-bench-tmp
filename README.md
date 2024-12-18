# Chart Document VQA Evaluation

## Setup
```
$ pip install -r requirements.txt
```
Run the script to install requirements

## Extract Contents
Run the following script to extract parsing results from various document parsing tools:  
```
$ python extract_contents.py \
    --image_dir <path_to_image_directory> \
    --model <choose from 'llamaparse', 'sonnet', 'gpt4o', 'upstage'> \
    --save_dir <directory_to_save_results>

```
Replace the placeholders (`<path_to_image_directory>` and `<directory_to_save_results>`) with the appropriate paths.  
Select the desired parsing model by specifying one of the available options (`'llamaparse'`, `'sonnet'`, `'gpt4o'`, or `'upstage'`).

## Run QA Evaluation
Run the following script to evaluate the extracted QA results:  
```
$ python run_qa_eval.py \
    --content_path <path_to_extracted_content> \
    --save_basepath <path_to_save_results>

```
Replace `<path_to_extracted_content>` and `<path_to_save_results>` with the appropriate paths.  
This script processes and evaluates the quality of the extracted QA content.
