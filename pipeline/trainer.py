import os
import json
import logging
import subprocess
import threading
from datetime import datetime
from app import db
from models import TrainingJob, PipelineStatus
from config import Config

# Try to import datasets and transformers, fallback if not available
try:
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_from_disk = None

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    transformers = None

class LLMTrainer:
    """
    LLM Fine-tuning trainer that works with HuggingFace datasets
    """
    
    def __init__(self):
        self.output_dir = Config.MODELS_DIR
        self._update_status('idle', 'Trainer initialized')
        
    def _update_status(self, status, message=None):
        """Update the trainer status in database"""
        try:
            trainer_status = PipelineStatus.query.filter_by(component='trainer').first()
            if not trainer_status:
                trainer_status = PipelineStatus(component='trainer')
                db.session.add(trainer_status)
            
            trainer_status.status = status
            trainer_status.last_activity = datetime.utcnow()
            if message:
                trainer_status.message = message
            
            db.session.commit()
        except Exception as e:
            logging.error(f"Failed to update trainer status: {e}")
    
    def start_training(self, dataset_path, epochs=None, batch_size=None, learning_rate=None):
        """
        Start LLM fine-tuning training
        """
        epochs = epochs or Config.DEFAULT_EPOCHS
        batch_size = batch_size or Config.DEFAULT_BATCH_SIZE
        learning_rate = learning_rate or Config.DEFAULT_LEARNING_RATE
        
        try:
            # Check if training dependencies are available
            if not HAS_DATASETS or not HAS_TRANSFORMERS:
                raise ValueError("Training dependencies not available. Please install 'datasets' and 'transformers' packages.")
            
            # Create training job record
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_output_path = os.path.join(self.output_dir, f'model_{timestamp}')
            
            # Try to get sample count from dataset
            total_samples = 0
            try:
                # Try to load dataset to get sample count
                if os.path.exists(os.path.join(dataset_path, 'dataset.json')):
                    with open(os.path.join(dataset_path, 'dataset.json'), 'r') as f:
                        data = json.load(f)
                        total_samples = len(data)
                elif HAS_DATASETS:
                    dataset = load_from_disk(dataset_path)
                    total_samples = len(dataset)
            except Exception as e:
                logging.warning(f"Could not determine dataset size: {e}")
            
            training_job = TrainingJob(
                status='pending',
                dataset_path=dataset_path,
                model_output_path=model_output_path,
                total_samples=total_samples,
                epochs=epochs
            )
            
            db.session.add(training_job)
            db.session.commit()
            
            job_id = training_job.id
            
            # Start training in background thread
            thread = threading.Thread(
                target=self._run_training,
                args=(job_id, dataset_path, model_output_path, epochs, batch_size, learning_rate)
            )
            thread.daemon = True
            thread.start()
            
            logging.info(f"Training job {job_id} started")
            return job_id
            
        except Exception as e:
            error_msg = f"Failed to start training: {e}"
            self._update_status('error', error_msg)
            logging.error(error_msg)
            raise
    
    def _run_training(self, job_id, dataset_path, model_output_path, epochs, batch_size, learning_rate):
        """
        Run the actual training process
        """
        try:
            self._update_status('running', f'Training job {job_id}')
            
            # Update job status
            job = TrainingJob.query.get(job_id)
            job.status = 'running'
            db.session.commit()
            
            # Create output directory
            os.makedirs(model_output_path, exist_ok=True)
            
            # Prepare training script
            training_script = self._create_training_script(
                dataset_path, model_output_path, epochs, batch_size, learning_rate
            )
            
            script_path = os.path.join(model_output_path, 'train.py')
            with open(script_path, 'w') as f:
                f.write(training_script)
            
            # Run training
            log_output = []
            process = subprocess.Popen(
                ['python', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=model_output_path
            )
            
            # Stream output
            for line in process.stdout:
                log_output.append(line)
                logging.info(f"Training {job_id}: {line.strip()}")
                
                # Update job with latest output
                job = TrainingJob.query.get(job_id)
                job.log_output = ''.join(log_output[-100:])  # Keep last 100 lines
                db.session.commit()
            
            process.wait()
            
            # Update final status
            job = TrainingJob.query.get(job_id)
            if process.returncode == 0:
                job.status = 'completed'
                job.log_output = ''.join(log_output)
                self._update_status('idle', f'Training job {job_id} completed')
                logging.info(f"Training job {job_id} completed successfully")
            else:
                job.status = 'failed'
                job.error_message = f"Training failed with return code {process.returncode}"
                job.log_output = ''.join(log_output)
                self._update_status('idle', f'Training job {job_id} failed')
                logging.error(f"Training job {job_id} failed")
            
            db.session.commit()
            
        except Exception as e:
            error_msg = f"Training job {job_id} failed: {e}"
            logging.error(error_msg)
            
            # Update job status
            try:
                job = TrainingJob.query.get(job_id)
                job.status = 'failed'
                job.error_message = str(e)
                db.session.commit()
            except:
                pass
            
            self._update_status('error', error_msg)
    
    def _create_training_script(self, dataset_path, output_path, epochs, batch_size, learning_rate):
        """
        Create a training script for fine-tuning
        """
        script = f'''
import os
import json
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

def main():
    print("Loading dataset...")
    dataset = load_from_disk("{dataset_path}")
    
    print("Loading model and tokenizer...")
    model_name = "{Config.HF_MODEL_NAME}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Preprocessing dataset...")
    def preprocess_function(examples):
        # Combine input and output for language modeling
        texts = []
        for i in range(len(examples['input'])):
            text = f"User: {{examples['input'][i]}}\\nAssistant: {{examples['output'][i]}}\\n"
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="{output_path}",
        overwrite_output_dir=True,
        num_train_epochs={epochs},
        per_device_train_batch_size={batch_size},
        learning_rate={learning_rate},
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("{output_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''
        return script
    
    def get_job_status(self, job_id):
        """Get status of a training job"""
        job = TrainingJob.query.get(job_id)
        if not job:
            return None
        
        return {
            'id': job.id,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'dataset_path': job.dataset_path,
            'model_output_path': job.model_output_path,
            'total_samples': job.total_samples,
            'epochs': job.epochs,
            'log_output': job.log_output,
            'error_message': job.error_message
        }
