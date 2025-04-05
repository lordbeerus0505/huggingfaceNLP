# Fine Tuning Large Language Models
# 11.1 Supervised Fine Tuning
Can fine tune LLMs on a broad range of tasks simultaneously using a process called `Supervised Fine Tuning` (SFT). Most LLMs people interact with have gone through SFT to make them more helpful and aligned with human preferences.

# 11.2 Chat Templates
Includes knowing how to properly format your conversation so as to get the best results from your model. Similar to prompt engineering.

## Model Types and Templates

### Base Models vs Instruct Models
>Base model is trained on raw text data to `predict the next token`. 

>Instruct model is fine-tuned to specifically `follow instructions and engage in conversations`.

Instruct tuned models are trained to follow a specific conversational structure making them suitable for chatbots. Instruct models can perform some agent like operations. 

ChatML or other template formats structure conversations with a clear indicator of roles - system, user, assistent.

### Common Template formats

Lets use this as the conversation structure -
```
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
```
This is what we use in SmolLM2 and Qwen 2
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
What's the weather?<|im_start|>assistant
```
So as mentioned above, there are 3 roles; the system describes to the assistant on what it should do. The assistant responds to user's messages.

Mistral uses this instead
```
<s>[INST] You are a helpful assistant. [/INST]
Hi! How can I help you today?</s>
[INST] Hello! [/INST]
```

Refer [this](https://huggingface.co/learn/nlp-course/chapter11/2?fw=pt) for specific differences between how many common models use these tags.

All of these can be handled automatically with the Transformers library -
```python
from transformers import AutoTokenizer

# These will use different templates automatically
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Each will format according to its model's template
mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)
```
So basically - use a role - content pair to describe what the messages are and the model's tokenizer will figure out how to format this as per the model's standards.

### Advanced features
Chat Templates can handle more advanced tasks such as tool use, multimodal inputs, function calling, multi-turn context.
Each of these would have their own variations on token usage.

For multimodal - this could mean content being a list of different types - text/image/audio. For tool usage - it could be complex like this 
```
messages = [
    {
        "role": "system",
        "content": "You are an AI assistant that can use tools. Available tools: calculator, weather_api",
    },
    {"role": "user", "content": "What's 123 * 456 and is it raining in Paris?"},
    {
        "role": "assistant",
        "content": "Let me help you with that.",
        "tool_calls": [
            {
                "tool": "calculator",
                "parameters": {"operation": "multiply", "x": 123, "y": 456},
            },
            {"tool": "weather_api", "parameters": {"city": "Paris", "country": "France"}},
        ],
    },
    {"role": "tool", "tool_name": "calculator", "content": "56088"},
    {
        "role": "tool",
        "tool_name": "weather_api",
        "content": "{'condition': 'rain', 'temperature': 15}",
    },
]
```

## Hands on with HFSmolLM2
Look at the notebook for this section. The `trl` library has a `setup_chat_format` method that you pass the model and tokenizer into and it returns them in the chat format needed.

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
```
Now if we define messages as so -
```python
# Define messages for SmolLM2
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {
        "role": "assistant",
        "content": "I'm doing well, thank you! How can I assist you today?",
    },
]
```
We can convert these into the chat template format using the tokenizer directly -
```python
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

print("Conversation with template:", input_text)
```
With output:
```
Conversation with template: <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you! How can I assist you today?<|im_end|>
```
If `tokenize=True`, it will return numbers as tokens. Similarly, adding the field `add_generation_prompt = True` would add `<|im_start|>assistant` at the end. 
We can also decode the tokenized inputs as discussed before if we set tokenize=True - `tokenizer.decode(token_ids=input_text)`

See the hands on exercise in the notebook

# 11.3 Supervised Fine-Tuning

Used to adapt pre-trained language models to follow instructions, engage in dialogue and use specific output formats. SFTs help transform pretrained models into assistants.

## When to use SFT
Would using an existing `instruction-tuned` model suffice? Only if not should you use SFT since its expensive to perform. 
> Need performance beyond what Prompting can offer. 

Do you need SFT? Only if Template Control and Domain Adaptation required.

### Template Control
SFT allow precise control over the model's output structure. Includes being able to
* Generate responses in a specific chat template format
* Follow strict output schemas
* Maintain consistent styling across responses

### Domain Adaptation
Align with domain specific requirements by:
* Teaching domain terminology and concepts ( so it doesnt stray away)
* Enforcing professional standards
* Handling technical queries appropriately
* Follow industry-specific guidelines

## Dataset Preparation
Requires a task specific dataset with input-output pairs having and input prompt; expected model response; additional context or metadata

## Training Configuration
There are several parameters that affect Training performance

1. Training Duration Parameters
`num_train_epochs` or `max_steps`. Large numbers could lead to overfitting
2. Batch Size Parameters
`per_device_train_batch_size` (determines memory usage and training stability), `gradient_accumulation_steps` (larger batch sizes)
3. Learning Rate Parameters
`learning_rate`, `warmup_ratio` (Portion of training used for learning rate warmup)
4. Monitoring Parameters
`logging_steps`(frequency of metric logging), `eval_steps`(freq of evaluating on validation data), `save_steps`(freq of checkpoint saves)

>"Start with conservative values and adjust based on monitoring: - Begin with 1-3 epochs - Use smaller batch sizes initially - Monitor validation metrics closely - Adjust learning rate if training is unstable"

## Implementation with TRL library

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# Start training
trainer.train()
```
Note the above requires a model and tokenizer to be defined. We can use the ones from the previous section -
```python
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
```
Since this uses a dataset with "messages" field, auto converts to ChatML format.

## Packing the dataset
SFT trainer allows multiple examples to be packed into the same input sequence. Requires setting `packing=True` in the constructor. Can disable packing for evaluation datasets with `eval_packing=False`.
```py
# Configure packing
training_args = SFTConfig(packing=True)

trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)

trainer.train()
```
Can use a custom format function to combine fields into a single input sequence.
```py
def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)
```

## Monitoring Training Progress

### Understanding Loss Patterns
3 distinct phases:
1. Initial Sharp Drop - rapid adaptation to the new distribution
2. Gradual Stabilization - Learning rate slows down as module fine tunes
3. Convergence - Loss stabilizes indicating training conclusion


### Metrics to monitor
Training loss, Validation loss, Learning rate progression, Gradient norms

## The path to convergence
Loss curve should gradually stabilize. Key indicator of training is small gap between training and validation loss implying model is learning to generalize as opposed to memorize.
See the graphs [here](https://huggingface.co/learn/llm-course/chapter11/3?fw=pt#monitoring-training-progress)

# 11.4 LoRA (Low-Rank Adaptation)

LoRA allows the fine tuning of LLMs with a small number of parameters. Reduces trainable parameters by 90% by adding and optimizing smaller matrices.

## Understand LoRA
Technique freezes pre-trained (base) model weights and injects trainable rank decomposition matrices into the model's layers. 
>LoRA decomposes the weight updates into smaller matrices through low-rank decomposition, significantly reducing the number of trainable parameters while maintaining model performance

Advantages of LoRA:
1. Memory Efficiency: 
    * Only adapter params stored in GPU memory
    * Base model weights are frozen - can load in lower precision
    * Fine tuning of large models on consumer GPUs

2. Training Features:
    * Native PEFT/LoRA integration with min setup
    * QLoRA has even better efficiency (quantized lora)

3. Adapter Management:
    * Adapter weight saving during checkpoints
    * Features to merge adapters into base models

## Loading LoRA with PEFT library
Parameter efficient fine tuning is a library by Hugging Face that allows you to easily add LoRA support. 
Can load adapaters onto pretrained models with `load_adaptor()`

```py
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

## Fine-tune LLM using trl and the SFTTrainer with LoRA
We can link SFTTrainer with LoRA using the LoRA Config from PeftConfig above. 

### LoRA configuration 
Key parameters are `rank` (r) - dimension of low-rank matrices used for weight updates. Typically between 4-32. `lora_alpha` scaling factor for LoRA layers ususally set to 2x rank value. `lora_dropout` - dropout probability for LoRA layers, typically 0.05 to 0.1. `bias` - controls bias terms like “none”, “all”, or “lora_only”. `target_modules` specifies which modules to apply LoRA to. Can be "all-linear" or specific.

### Using TRL with PEFT
```py
from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 6
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,  # LoRA configuration
    max_seq_length=max_seq_length,  # Maximum sequence length
    processing_class=tokenizer,
)
```

## Merging LoRA Adapters
You might want to merge the adapter weights back into the base model for easier after training with LoRA. 

>The merging process requires attention to memory management and precision. Since you’ll need to load both the base model and adapter weights simultaneously, ensure sufficient GPU/CPU memory is available. Using device_map="auto" in transformers will find the correct device for the model based on your hardware.

```py
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name", torch_dtype=torch.float16, device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model, "path/to/adapter", torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
merged_model = peft_model.merge_and_unload()

# If there are issues
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

# 11.5 Evaluation
Basic theory - read from [here](https://huggingface.co/learn/llm-course/chapter11/5?fw=pt#evaluation)
