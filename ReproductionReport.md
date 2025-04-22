# Training and Evaluation code for TinyAgent

#### 1. Created `data` folder in the main directory and downloaded train and test dataset files from huggingface with `wget https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset/resolve/main/testing_data.json` and `.../training_data.json`

#### Raw Dataset examination:
Each data sample in the dataset has these fields
- `input` - Clear User prompt without any annotations/context added
- `output` - list where there are two types of outputs `plan` and `join`, at first I didn't understand what was the `join` type for, but I think I have an idea about it now, will talk about it later, now focus on type `plan` which I would say is the main data
- `output.raw_input` - as I understood it's the full annotated input to the LM, with `'role': 'system'` and `'role': 'user'` where the first one is context for the LM, with ToolRag already used, and it contains some instructions that the LM must follow and tools that it needs to use for this task and the prompt is the same `input`
- `output.raw_output` - the response the LM should return
- `output.parsed_output` - the tools are listed one by one with their arguments, which I did not use for the fine-tuning because the tools are already present in the prompt

#### Dataset preprocessing
Added `preprocess_dataset` function in `utils/data_utils.py` for simple dataset preprocessing, creates a 'cleaner' version of the dataset that contains the instruction, user prompt and target result. This method is being called in the `train.py` file.

#### Training
*We applied LoRA (Hu et al., 2021) to fine-tune the models for 3 epochs using a learning rate of 7e-5 over the 80K training examples, and selected the best checkpoint based on validation performance. After fine-tuning, the 1.1B model improved the success rate from 12.71% to 78.89%*

**NOTE:** they say they trained on 80K examples, but there are 40K examples of type "plan", I think they also trained on type "join" which if I understood correctly just reevaluates the job it has done, and if it thinks there is something wrong, tries to "replan", but I didn't understand it at first so the results I got are attained on 40K set.
Looking at the base model [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct) that TinyAgent was fine-tuned on, I kept the same format of prompt.

Authors have most probably used batch size of 8 looking at the [config.json](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B/blob/main/config.json) file in their huggingface model page. Used adamw_8bit for optimizer for memory saving and 4-bit NF4 quantization.
I created 1K validation set from the 40K train set and trained the model for 3 epochs with the learning rate 7e-5 on L4 GPU(with 24GB memory) for about 33 hours, the final success rate on the separate 1K test set is 78.4% compared to their 78.89%.
I faced some GPU memory related problems during the training, batch size greater than 1 was running out of memory(because seq_length is quite big) even with the quantization settings applied, so I had to use gradient_accumulation. I think authors were using at least one A100 GPU with 40GB memory. 


#### Run
- Env `pip install -r requirements.txt`
- Training `python -m src.train.agent.train`
- Evaluation `python -m src.train.agent.eval`

What can be improved:

1. Using all 80K samples from Dataset and allowing the LM to "replan" so it can reevaluate itself and have a chance to fix itself.
2. Running the training for a little longer, maybe 4 epochs and then selecting the best checkpoint, because now with 3 epochs I see that there is some room for improvement and the model was still improving.
3. Using LoRA not only on attention modules, but also on mlp layers gate_proj, up_proj and down_proj.
4. At last maybe experimenting with LoRA rank 32 instead of current 16.