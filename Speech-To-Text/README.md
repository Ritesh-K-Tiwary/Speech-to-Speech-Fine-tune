Following steps are used to fine-tune Whisper model:
## Environment Setup
Install required python libraries from `requirements.txt` and login to using `write token` of your Huggingface account via:
```python
from huggingface_hub import notebook_login
notebook_login()
```
Define wer metric for evaluating output of your model:
```python
from datasets import load_metric
# Load the WER metric
wer_metric = load_metric("wer")
```
Import following libraries:
```python
import pandas as pd
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig
from dataclasses import dataclass
from typing import Any, Dict, List, Union
```
## Model Quantization
Set up quantization configuration and Load the model with quantization:
```python
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny",
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)

# Set the language to English
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny",language='en', task="transcribe")
forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")
model.config.forced_decoder_ids = forced_decoder_ids
```
Use Lora configuration:
```python

# Apply LoRA configuration with specified target modules
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ2SEQ_LM",
    inference_mode=False,
    target_modules="all-linear"
    # target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]  # Specifying common attention layer modules
)

# Add the adapter
# model = get_peft_model(model, peft_config)
model.add_adapter(peft_config)
```
## Data Preprocessing
```python
import glob
import os
import csv

# Path to the folder containing the CSV files
folder_path = '/home/arun/ritesh/practice/sayaliWork/processed_csv_folder_test'

# Create a list to hold the DataFrames
data1 = []

# Use glob to find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Loop through the list of CSV files and read each one into a DataFrame
for file in csv_files:
    df = pd.read_csv(file, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, skipinitialspace=True)
    data1.append(df)
```
```python

# Combine all DataFrames into a single DataFrame
combined_data1 = pd.concat(data1, ignore_index=True)

# Function to modify the 'audio' column
def modify_audio_path(path):
    return path.replace("processed_", "output_segments/")

# Apply the function to the 'audio' column
combined_data1['audio'] = combined_data1['audio'].apply(modify_audio_path)

# Function to check if the file exists
def file_exists(path):
    return os.path.exists(os.path.join('/home/arun/ritesh/practice/sayaliWork', path))

# Filter the DataFrame to include only rows where the file exists
combined_data1 = combined_data1[combined_data1['audio'].apply(file_exists)]

combined_data1
```
Load and Preprocess audio data and applying filter:
```python
# Function to load and process audio
def load_and_process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)  # Load and resample to 16000 Hz
        if len(y) == 0:  # Check for zero-length audio
            print(f"Skipping zero-length audio file: {file_path}")
            return None  # Return None for zero-length audio files
        mel_spec = processor.feature_extractor(y, sampling_rate=sr, return_tensors="pt").input_features[0]
        return mel_spec
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Apply the function to load and process audio, creating a new column
combined_data1["input_features"] = combined_data1["audio"].apply(load_and_process_audio)

# Remove rows where 'input_features' is None
combined_data1 = combined_data1[combined_data1['input_features'].notnull()].copy()

# Tokenize the text
combined_data1["labels"] = combined_data1["text"].apply(lambda x: processor.tokenizer(x).input_ids)

# Function to prepare the dataset
def prepare_dataset(row):
    return {"input_features": row["input_features"], "labels": row["labels"]}

# Apply the prepare_dataset function to each row of the DataFrame
processed_data = combined_data1.apply(prepare_dataset, axis=1)

# Convert the processed data to a list of dictionaries
processed_list = processed_data.tolist()
```
## Pretraining Process
Split Data:
```python
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
train_set, test_set = train_test_split(processed_list, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
```

Training Argument Setup
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Adjusted to simulate a larger batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=128,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    save_total_limit=None,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    overwrite_output_dir=True,  # Overwrite the output directory if it exists
)
```
Define Data Collator:
```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["input_features"] = batch["input_features"].to(torch.float16 if training_args.fp16 else torch.float32)
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)
```
Define Compute Metric:
```python
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```
Define Early Stopping:
```python
from transformers import EarlyStoppingCallback

# Define the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluation steps to wait for improvement
    early_stopping_threshold=0.0,  # Minimum improvement to qualify as improvement
)
```
## Training Setup
```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[early_stopping_callback]
)
```
Training:
```python
import warnings

# Suppress specific warnings related to torch.utils.checkpoint
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")


trainer.train()
```
## Save Model to Local Directory and Your Huggingface account:
```python
from transformers import WhisperConfig, WhisperTokenizer, AutoModelForSpeechSeq2Seq

# Load and save config.json
config = WhisperConfig.from_pretrained("openai/whisper-large-v3")
config.save_pretrained("./whisper-large-v3-quantized")

# Load and save tokenizer files
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
tokenizer.save_pretrained("./whisper-large-v3-quantized")

# After quantization, load the quantized model
model = AutoModelForSpeechSeq2Seq.from_pretrained("./whisper-large-v3-quantized")
model.save_pretrained('./whisper-large-v3-quantized-repo')

# Push all files to the Hugging Face Hub
from huggingface_hub import Repository
repo = Repository("./whisper-large-v3-quantized", clone_from="riteshkr/whisper-large-v3-quantized")
repo.push_to_hub(commit_message="Added quantized model, config, and tokenizer files")
```
## Inference
```python
from transformers import pipeline

model_id = "./whisper-large-v3-quantized"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)

def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]
transcribe_speech("output_segments/F_0050_10y9m_1/segment_2.wav")
```
