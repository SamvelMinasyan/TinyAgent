import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM,
)
from peft import LoraConfig, TaskType
import hydra
from omegaconf import DictConfig

from src.utils.data_utils import preprocess_dataset

torch.cuda.empty_cache()

@hydra.main(config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig) -> None:

    if not os.path.exists(cfg.train_data_json):
        preprocess_dataset("../../../data/training_data.json", cfg.train_data_json)
    if not os.path.exists(cfg.test_data_json):
        preprocess_dataset("../../../data/test_data.json", cfg.test_data_json) # test file is not used here, just preprocessing for the future
    # Load training and evaluation datasets
    train_eval_dataset = load_dataset(
        "json", data_files=cfg.train_data_json, split="train"
    )

    eval_dataset = train_eval_dataset.select(range(1000))
    train_dataset = train_eval_dataset.select(range(1000, len(train_eval_dataset)))

    # Transform each sample into a formatted prompt string
    def format_conversation(example):

        system_text = example["data"][0]["content"].strip() if example["data"][0]['role'] == "system" else ""
        user_text = example["data"][1]["content"].strip() if example["data"][1]['role'] == "user" else ""
        assistant_text = example["data"][2]["content"].strip() if example["data"][2]['role'] == "assistant" else ""

        # Build the prompt in the format as on model's huggingface page.
        full_prompt = (
            f"### Instruction:\n{system_text}\n\n"
            f"### Input:\n{user_text}\n\n"
            f"### Response:\n{assistant_text}<END_OF_PLAN>"
        )
        return {"text": full_prompt}

    train_dataset = train_dataset.map(format_conversation, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_conversation, remove_columns=eval_dataset.column_names)

    # Initialize tokenizer and prepare text templates
    token_engine = AutoTokenizer.from_pretrained(cfg.model_id)
    instr_template = token_engine.encode("<s>### Instruction:\n", add_special_tokens=False)
    resp_template = token_engine.encode("### Response:\n", add_special_tokens=False)[1:]

    # Setup data collator that masks out non-response tokens from the loss computation
    collate_fn = DataCollatorForCompletionOnlyLM(
        instruction_template=instr_template,
        response_template=resp_template,
        tokenizer=token_engine,
        mlm=False,
    )

    # Configure LoRA
    lora_params = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        lora_dropout=cfg.lora.dropout,
        bias="none",
    )

    # Define quantization settings (4-bit NF4)
    quant_settings = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the pretrained causal lm with quantization
    model_engine = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=quant_settings,
    )

    # Configure SFT training parameters
    sft_config = SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.batch.train,
        per_device_eval_batch_size=cfg.batch.eval,
        gradient_accumulation_steps=cfg.grad_acum_steps,
        learning_rate=cfg.lr,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        max_seq_length=cfg.max_seq_length,
        warmup_steps=cfg.warmup_steps,
        optim="adamw_8bit",
        logging_strategy=cfg.logging_strategy,
        logging_steps=cfg.log_steps
    )

    # Create and initialize the SFT trainer with all components
    trainer = SFTTrainer(
        model=model_engine,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=lora_params,
        processing_class=token_engine,
    )

    trainer.train()
    trainer.save_model(cfg.out_dir)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()