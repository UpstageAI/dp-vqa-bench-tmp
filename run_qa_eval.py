import os
import json
import openai
import argparse

from glob import glob

OPENAI_API_KEY = ""


def ask_question(prompt, model):
    openai.api_key = OPENAI_API_KEY
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions given the content within the document."
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer


def read_question_and_answer(basepath, filename):
    question_paths = glob(os.path.join(basepath, filename + "*"))
    if len(question_paths) == 1:
        question_path = question_paths[0]
        with open(question_path, "r") as f:
            question_and_answer = json.load(f)

            question_only = []
            answer_only = []
            for question_elem in question_and_answer:
                # remove "short_answer" key from the question
                answer = question_elem.pop("short_answer")

                question_elem["short_answer"] = ""
                question_only.append(question_elem)
                answer_only.append(answer)

            question_str = json.dumps(question_only, indent=2)
            answer_str = json.dumps(answer_only, indent=2)

    else:
        print("Error: Multiple question files found for: ", filename)
        exit(0)

    return question_str, answer_str


def prepare_question_prompt(
    template, content, question,
    content_key="<CONTENTSECTION>",
    question_key="<QUESTIONSECTION>"
):

    template = template.replace(content_key, content)
    template = template.replace(question_key, question)

    return template


def reformat_answer(answer):
    if "```" in answer and "json" in answer:
        answer = answer[7:]
        answer = answer[:-3]

    answer = json.loads(answer)
    return answer


def prepare_eval_prompt(template, ref_answer, model_answer):
    ref_answer = json.loads(ref_answer)
    ref_answer = str([elem for elem in ref_answer])
    model_answers = str([elem["short_answer"] for elem in model_answer])

    template = template.replace("<REFERENCEANSWERSECTION>", ref_answer)
    template = template.replace("<MODELANSWERSECTION>", model_answers)

    return template, ref_answer, model_answers


def evaluate(evaluate_prompt, model):
    openai.api_key = OPENAI_API_KEY
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that evaluates list of model answers given the reference answer."
            },
            {
                "role": "user", "content": evaluate_prompt
            }
        ]
    )
    scores = response.choices[0].message.content
    return scores


def reformat_scores(questions, ref_str, model_str, scores):
    question_text = ""
    for idx, question_elem in enumerate(json.loads(questions)):
        question_text += str(idx+1) + ". " + question_elem["question"] + "\n"

    if "```" in scores and "list" in scores:
        scores = scores[7:]
        scores = scores[:-3]

    scores = list(json.loads(scores))

    scores_text = ""
    scores_text += "Questions: \n" + question_text
    scores_text += "\n\nReference Answers: \n" + ref_str
    scores_text += "\n\nModel Answers: \n" + model_str

    scores_text += "\n\nScores: \n"
    scores_text += str(scores)

    return scores, scores_text


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--content_path", type=str, default="", required=True)
    argparser.add_argument("--question_basepath", type=str, default="dataset/questions")
    argparser.add_argument("--question_template_path", type=str, default="dataset/templates/question_template.txt")
    argparser.add_argument("--evaluate_template_path", type=str, default="dataset/templates/evaluate_template.txt")
    argparser.add_argument("--answering_model", type=str, default="gpt-4o")
    argparser.add_argument("--evaluate_model", type=str, default="gpt-4o")
    argparser.add_argument("--save_basepath", type=str, default="results/")
    args = argparser.parse_args()

    file_paths = glob(os.path.join(args.content_path, "*.txt"))

    counter = 0
    all_scores = []
    for file_path in file_paths:
        if "time.txt" in file_path:
            continue
        print("({}/{}) Processing file: ".format(counter+1, len(file_paths)), file_path)

        basename = '.'.join(os.path.basename(file_path).split(".")[:-1])

        save_basepath = os.path.join(args.save_basepath, basename)
        if not os.path.exists(save_basepath):
            os.makedirs(save_basepath)

        with open(args.question_template_path, "r") as f:
            question_template = f.read()

        with open(args.evaluate_template_path, "r") as f:
            eval_template = f.read()

        with open(file_path, "r") as f:
            content = f.read()

        question, reference_answers = read_question_and_answer(args.question_basepath, basename)
        question_prompt = prepare_question_prompt(question_template, content, question)
        answer = ask_question(question_prompt, args.answering_model)

        json_answer = reformat_answer(answer)

        prompt_save_path = os.path.join(save_basepath, "prompt.txt")
        with open(prompt_save_path, "w") as f:
            f.write(question_prompt)

        evaluate_prompt, reference_list, answer_list = prepare_eval_prompt(
            eval_template, reference_answers, json_answer
        )
        evaluate_result = evaluate(evaluate_prompt, args.evaluate_model)
        scores_list, scores_text = reformat_scores(
            question, reference_list, answer_list, evaluate_result
        )

        all_scores += scores_list

        eval_save_path = os.path.join(save_basepath, "scores.txt")
        with open(eval_save_path, "w") as f:
            f.write(scores_text)

        counter += 1

    avg_score = sum(all_scores) / len(all_scores)
    total_questions = len(all_scores)

    score_path = os.path.join(args.save_basepath, "final_score.txt")
    with open(score_path, "w") as f:
        f.write("Mean score: {:.4f}\n".format(avg_score))
        f.write("Total number of questions: {}".format(total_questions))

    print("Mean score: {:.4f}".format(avg_score))
    print("Total number of questions: ".format(total_questions))
