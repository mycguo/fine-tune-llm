# https://www.youtube.com/watch?v=Vg3dS-NLUT4
import os
import torch
import streamlit as st
import warnings
from datasets import load_dataset

# Suppress invalid escape sequence warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="peft")

# Set memory management for MPS
if torch.backends.mps.is_available():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    torch.mps.empty_cache()

# Handle imports with error checking
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        pipeline,
        logging,
    )
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
except ImportError as e:
    st.error(f"""
    Error importing required packages. Please install them using:
    ```
    pip install transformers peft trl
    ```
    Error details: {str(e)}
    """)
    st.stop()

st.set_page_config(page_title="LLM Fine-tuning", layout="wide")

st.title("LLM Fine-tuning Interface")

# Model and dataset configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

# Training parameters
with st.sidebar:
    st.header("Training Parameters")
    lora_r = st.slider("LoRA attention dimension", 8, 16, 8)
    lora_alpha = st.slider("LoRA alpha", 8, 32, 16)
    lora_dropout = st.slider("LoRA dropout", 0.0, 0.5, 0.1)
    num_train_epochs = st.slider("Number of epochs", 1, 10, 1)
    learning_rate = st.number_input("Learning rate", 1e-5, 1e-3, 2e-4, format="%.2e")
    batch_size = st.slider("Batch size", 1, 4, 1)  # Reduced max batch size
    gradient_accumulation_steps = st.slider("Gradient accumulation steps", 1, 16, 4)

def initialize_model():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        st.info("Using MPS device")
        compute_dtype = torch.float16
        # Enable memory efficient attention
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        st.info("Using CUDA device")
        compute_dtype = torch.float16
    else:
        device = torch.device("cpu")
        st.info("Using CPU device")
        compute_dtype = torch.float32

    # Load base model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def train_model():
    with st.spinner("Initializing model and tokenizer..."):
        model, tokenizer = initialize_model()
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset = load_dataset(dataset_name, split="train")
        st.info(f"Dataset loaded with {len(dataset)} examples")

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training arguments
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=0,
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    # Initialize trainer
    with st.spinner("Setting up trainer..."):
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_arguments,
        )

    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    def training_callback(progress):
        progress_bar.progress(progress)
        status_text.text(f"Training progress: {progress:.0f}%")

    # Start training
    with st.spinner("Training in progress..."):
        trainer.train()
        trainer.model.save_pretrained(new_model)
    
    st.success("Training completed! Model saved successfully.")

# Main interface
if st.button("Start Training", type="primary"):
    train_model()

