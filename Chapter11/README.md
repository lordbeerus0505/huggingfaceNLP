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
