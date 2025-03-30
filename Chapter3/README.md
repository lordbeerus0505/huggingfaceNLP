# 3.2 Processing the data
This is how to train a sequence classifier on one batch -
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new - explained later.
from torch.optim import AdamW
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```
This is a small setup with just 2 datasets so the training does not mean much. Instead we will use the MRPC dataset. 
> MRPC is a dataset that is part of the GLUE benchmark to measure performance of ML models across 10 classification tasks.
## Loading a dataset from HF Hub
Can load this dataset as 
```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```
Which will output the distribution across train test and validation
```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

Access the train dataset via
```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

See the features with `raw_train_dataset.features`
## Preprocessing a dataset
Preprocessing requires converting the text to numbers the model can make sense of - Done with a tokenizer. We can feed the tokenizer one sentence or a list of sentences like below - all the `sentence1` of tokenizer are sent together to the tokenizer and so are `sentence2`
```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```
The tokenizer also allows you to send pairs like how BERT expects:
```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```
Returns the input_ids, token_type_ids and attention_mask
> token_type_ids is a list of [0s or 1s] in this case where 0 is the first sentence and 1 is the second sentence. Its binary so if you give 10 sentences to tokenizer, 2-10 will all be 1.

Since there are no padding tokens attention_mask is all 1.
```python
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```
Can decode the tokens back to words -
```python
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```
And if we match these with the token_type_ids.
```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```
> DistilBERT does not return token_type_ids. Since BERT is pretrained with token_type_ids, it knows what to do with them.

Can pass the tokenizer the truncate and padding options when passing 2 inputs - this is fine except, it requires the ability to load the entire dataset into RAM. When using raw_datasets directly, we are loading from disk dynamically.
```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
To keep the data as a dataset - use `Dataset.map()`
The map method applies a function on each element of the dataset - 
```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
Similar to generators and yield.

We can pass to this function a batch of example, not just 1. That way it will run fast, load a bunch of them into memory and we can process that batch.

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

`tokenized_datasets` have extra fields compared to raw_datasets - input_ids, token_type_ids, attention_mask - the same 3 fields when you call `tokenizer()` since these are all `keys` in the dictionary returned by the preprocessor - in this case the response `tokenize_function`.

> Can you multiprocessing module, pass num_proc argument and speed this up even more.

## Dynamic Padding

> Collate Function is responsible for putting together samples inside a batch.

Can dynamically pad using this module.
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Lets use the first 8 inputs
samples = tokenized_datasets["train"][:8]
# We are only collecting the inputs_idx, token_type_ids, attention_mask and labels since the others have strings or useless info.
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
# The max is 67 - so all should be padded to that -
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```
Output:
```
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
```
Notice how they are all of size 8x67
>  "Dynamic padding means the samples in this batch should all be padded to a length of 67, the maximum length inside the batch. Without dynamic padding, all of the samples would have to be padded to the maximum length in the whole dataset, or the maximum length the model can accept"

# 3.3 Fine Tuning a model with the Trainer API
Trainer class in transformers allows fine-tuning of any pretrained models on your dataset.
Summarized code from earlier
```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Training
`TrainingArguments` class will contain all hyperparameters the Trainer will use for training and evaluation.

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
Now pass to the trainer all objects constructure
```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator, # by default set to DataCollatorWithPadding
    tokenizer=tokenizer,
)
```
To train call - requires setting up wandb account to get the API key.
```python
trainer.train()
```

## Evaluation

To get predictions use `Trainer.predict()`
```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```
The output of predict() is another named tuple with 3 fields: `prediction`, `label_ids` and `metrics`. 
There are 408 elements in the dataset so the predictions array is of shape 408x2 since it returns logits (probability of true and false). To convert to readable predictions - 
```python
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```
To compare these predictions to labels can use hugging face `Evaluate`
```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```
Returns accurace and f1-score.
Our final `compute_metrics` method.

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

The new trainer with compute_metrics incorporated:
```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```
Now running `trainer.train()` will also give you scores of evaluation.

# 3.4 A full training

How to do a full training loop but without a Trainer from 3.3?
Lets go back to the post - pre-processing step -
```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
Tokenization + Padding is done, now we fine tune.

## Prepare for training

Dataloaders are used to iterate over batches. Before using these, need some more postprocessing that `Trainer` did for us - 
* Remove columns corresponding to values the model does not exoect (strings like sentence1, sentence2)
* Rename column `label` to `labels` because pyTorch needs that
* Set format of tensors to be that of PyTorch

We can do this with the tokenized dataset directly -
```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names # Only attention_mask, labels, input_ids and token_type_ids left.
```

With this, we can define the Dataloader to deal with batches of a specified size -
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```
`shuffle=True` implies the contents change on each run and they arent following the defined order, which could negatively affect the model's learning capacity.

Now instantiate the model like before -
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Now we will setup the optimizer and learning rate scheduler -
```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```
> Adam or Adaptive Moment Estimation is an optimization algorithm that dynamically adjusts learning rate for each parameter, without using a single global learning rate. It combines the strength of Stochastic Gradient Descent with Momentum (A momentum variable is used in calculating how much weights are to be adjusted, where the momentum changes recursively based on the gradient of the loss function) and RMSprop (Root Mean Squared Propagation) to update model parameters.
Ref: https://www.geeksforgeeks.org/adam-optimizer/. AdamW uses Decoupled Weight Decay Regularization.

Learning rate is a linear decay from 5*10^-5 to 0 as defined in the lr_scheduler below -
```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```
each step of the 1377 steps will linearly decrease the learning rate till it comes down to 0 from 5e-5.

Next, we specify the device to use (GPU for colab) and then train - 
```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad() # Reset gradients before next iteration.
        progress_bar.update(1)
```
## The evaluation loop
Can handle batches with the metric.compute() method as well
```python
# 1 batch
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```
## Supercharging with Accelerate
We can use the `Accelerator` module, with `accelerator = Accelerator()` used to setup a distributed environment. `accelerate.prepare()` wraps objects in the proper container to make sure your distributed training works as intended. The loss propagation is now done with `accelerate.backward(loss)`
```python
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
