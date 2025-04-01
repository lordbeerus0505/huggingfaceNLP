# Sharing Models and Tokenizers
# 4.2 Using Pretrained Models

Can easily load a mask filling model like
```python
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
```
If we load this pipeline for something else instead like text-classification, the results would not make sence since the head of `camembert-base` is not relevant.

Can instantiate using model architecture directly like
```python
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```
but its better to use uniform AutoClasses since they are architecture agnostic.
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

# 4.3 Sharing pretrained models (PyTorch)

Can use the following to share models -
* Using the push_to_hub API
* Using the huggingface_hub Python library
* Using the web interface

Can login using
```python
from huggingface_hub import notebook_login

notebook_login()
```
or use `huggingface-cli login` in the terminal.
Auth token is stored in cache after this step.

Setting `push_to_hub` = True will start pushing to hub - when trainer.train() is called. Added to the hyperparameters in TrainingArguments
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
```
Can specify a `hub_model_id` for the output directory. Once pushed, a model card is created that has the `Training Hyperparameters` and the `Training results`.

## Using the push_to_hub API
> Using the `push_to_hub()` method, you can pass a model, tokenizer or config object directly to the hub at any point.

Say you use this model -
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Once done with training it - push to hub -
```python
model.push_to_hub("dummy-model") # Push model
tokenizer.push_to_hub("dummy-model") # Push tokenizer
```

## Using the huggingface_hub Python library
Read from [here](https://huggingface.co/learn/nlp-course/chapter4/3?fw=pt#using-the-huggingfacehub-python-library)

Nothing too technical to be explained.

## Using the web interface
Read from [here](https://huggingface.co/learn/nlp-course/chapter4/3?fw=pt#using-the-huggingfacehub-python-library)
