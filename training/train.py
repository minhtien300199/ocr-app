import os
import torch
from torch.utils.data import DataLoader
import easyocr
from config.training_config import TRAINING_CONFIG, DATASET_CONFIG, MODEL_PATHS
from PIL import Image
import numpy as np
from tqdm import tqdm

class CustomOCRTrainer:
    def __init__(self):
        self.device = torch.device(f"cuda:{TRAINING_CONFIG['gpu_id']}" 
                                 if torch.cuda.is_available() and TRAINING_CONFIG['gpu_id'] >= 0 
                                 else "cpu")
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.epochs = TRAINING_CONFIG['epochs']
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(DATASET_CONFIG['languages'], 
                                   gpu=TRAINING_CONFIG['gpu_id'] >= 0)
        
        # Create model save directory
        os.makedirs(MODEL_PATHS['save_dir'], exist_ok=True)

    def prepare_dataset(self):
        """Prepare dataset for training"""
        train_data = []
        
        # Load training data
        train_dir = DATASET_CONFIG['train_data_dir']
        for lang_dir in os.listdir(train_dir):
            lang_path = os.path.join(train_dir, lang_dir)
            if os.path.isdir(lang_path):
                for img_file in os.listdir(lang_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(lang_path, img_file)
                        # Get ground truth from filename (assuming filename format: text_groundtruth.jpg)
                        ground_truth = img_file.split('_')[0]
                        train_data.append((img_path, ground_truth))
        
        return train_data

    def train_model(self, train_data):
        """Train the OCR model"""
        print("Starting training...")
        
        # Get the recognition model from EasyOCR
        recognition_model = self.reader.recognizer.model
        recognition_model.train()
        recognition_model.to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(recognition_model.parameters(), 
                                   lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_data, desc=f'Epoch {epoch+1}/{self.epochs}')
            
            for img_path, ground_truth in progress_bar:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img)
                    
                    # Get model predictions
                    result = self.reader.readtext(img)
                    
                    # Calculate loss (simplified)
                    loss = 0
                    if result:
                        predicted_text = result[0][1]
                        loss = self.calculate_loss(predicted_text, ground_truth)
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': total_loss/(progress_bar.n+1)})
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
            
            # Save model after each epoch
            self.save_model(recognition_model, epoch)
            
            print(f"Epoch {epoch+1} completed. Average loss: {total_loss/len(train_data)}")

    def calculate_loss(self, predicted_text, ground_truth):
        """Calculate loss between predicted text and ground truth"""
        # Implement your custom loss function here
        # This is a simplified example
        return torch.nn.functional.cross_entropy(
            torch.tensor([ord(c) for c in predicted_text]), 
            torch.tensor([ord(c) for c in ground_truth])
        )

    def save_model(self, model, epoch):
        """Save the trained model"""
        save_path = os.path.join(MODEL_PATHS['save_dir'], 
                                f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

def main():
    trainer = CustomOCRTrainer()
    train_data = trainer.prepare_dataset()
    trainer.train_model(train_data)

if __name__ == "__main__":
    main() 