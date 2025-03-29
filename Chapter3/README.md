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

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```