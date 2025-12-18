import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_folder_name = "dataset_processed_Llama-3.1-8B-Instruct" 
dataset_path = os.path.join(script_dir, dataset_folder_name)

output_dir = os.path.join(script_dir, "llama3-aymara-yatichiri-adapter")
model_name = "meta-llama/Llama-3.1-8B-Instruct"


print(f"Buscando dataset PRE-PROCESADO en: {dataset_path}")
if not os.path.exists(dataset_path):
    print(f"ERROR CRÍTICO: No existe la carpeta {dataset_path}")
    print("Por favor, revisa el nombre de la carpeta generada en el paso anterior.")
    exit()

print("Dataset encontrado. Cargando...")
dataset = load_from_disk(dataset_path)
print(f"Filas de entrenamiento listas: {len(dataset)}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
)


print("Cargando modelo base Llama 3.1...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False, 
    
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
    lora_alpha=32,     
    lora_dropout=0.05,
    r=32,              
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

model = get_peft_model(model, peft_config)
print("\n--- Parámetros entrenables ---")
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,                  
    per_device_train_batch_size=4,       
    gradient_accumulation_steps=4,       
    optim="paged_adamw_32bit",
    save_steps=100,                      
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True,                           
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    gradient_checkpointing=True,         
    report_to="none"                     
)

print("\nIniciando entrenamiento (Yatichiri Mode)...")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,      
    processing_class=tokenizer, 
    args=training_args,
)


trainer.train()

print(f"Entrenamiento finalizado. Guardando adaptador en: {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("¡Listo! Ahora tienes un modelo que prioriza el Aymara.")