from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel

from src.utils.plan_utils import get_parsed_planner_output_from_raw
from src.utils.graph_utils import compare_graphs_with_success_rate, build_graph


def evaluate():

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    checkpoint_dir = "outputs/2025-04-18/21-14-50/model/checkpoint-14634"
    model_name = "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"

    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_dataset = load_dataset(
        "json", data_files="data/test_data_preprocessed.jsonl", split="train"
    )

    metrics = []
    for sample in tqdm(test_dataset):
        msgs = sample["data"]
        prompt = (
            f"### Instruction:\n{msgs[0]['content']}\n\n"
            f"### Input:\n{msgs[1]['content']}\n\n"
            "### Response:\n"
        )
        target_text = msgs[-1]["content"]
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        """ 
        Calculating the length of the target_response tokens for dynamic max_new_tokens setting, 
        initially I was using the max length of the target in the database,
        but then understood that setting it dynamically will drastically improve the speed of evaluation(from 7-8 hours to just 2 hours)
        
        + 10 in the end is the length of the '<END_OF_PLAN>' in tokens (8) and + 2 as a buffer just in case
        """

        target_tokens_length = len(tokenizer(target_text, add_special_tokens=False)['input_ids']) + 10

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=target_tokens_length) # 264 max length (+ buffer) in the test dataset

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].split("<END_OF_PLAN>")[0]


        target_plan = get_parsed_planner_output_from_raw(target_text)
        generated_plan = get_parsed_planner_output_from_raw(response)
        target_graph = build_graph(target_plan)
        generated_graph = build_graph(generated_plan)
        success_rate = compare_graphs_with_success_rate(target_graph, generated_graph)
        metrics.append(success_rate)

    print("Mean success rate:", np.mean(metrics))


if __name__ == '__main__':
    evaluate()
