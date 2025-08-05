import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BloodGroupVisualizer:
    def __init__(self, dataset_path='dataset_blood_group'):
        self.dataset_path = dataset_path
        self.blood_groups = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(self.blood_groups)))
        
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training and validation loss/accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss', color='#FF6B6B', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='#4ECDC4', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history.history['accuracy'], label='Training Accuracy', color='#45B7D1', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#96CEB4', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix with heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.blood_groups, 
                    yticklabels=self.blood_groups)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_data_distribution(self, save_path='data_distribution.png'):
        """Plot distribution of samples across blood groups"""
        counts = []
        for blood_group in self.blood_groups:
            group_path = os.path.join(self.dataset_path, blood_group)
            if os.path.exists(group_path):
                count = len([f for f in os.listdir(group_path) if f.endswith('.BMP')])
                counts.append(count)
            else:
                counts.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.blood_groups, counts, color=self.colors, alpha=0.8)
        plt.title('Dataset Distribution by Blood Group', fontsize=16, fontweight='bold')
        plt.xlabel('Blood Group', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def display_sample_images(self, samples_per_group=3, save_path='sample_images.png'):
        """Display sample images from each blood group"""
        fig, axes = plt.subplots(len(self.blood_groups), samples_per_group, 
                                figsize=(15, 20))
        
        for i, blood_group in enumerate(self.blood_groups):
            group_path = os.path.join(self.dataset_path, blood_group)
            if os.path.exists(group_path):
                image_files = [f for f in os.listdir(group_path) if f.endswith('.BMP')]
                
                for j in range(samples_per_group):
                    if j < len(image_files):
                        img_path = os.path.join(group_path, image_files[j])
                        img = Image.open(img_path)
                        img = img.resize((64, 64))
                        
                        if samples_per_group == 1:
                            ax = axes[i]
                        else:
                            ax = axes[i, j]
                        
                        ax.imshow(img, cmap='gray')
                        ax.set_title(f'{blood_group}', fontsize=10, fontweight='bold')
                        ax.axis('off')
                    else:
                        if samples_per_group == 1:
                            ax = axes[i]
                        else:
                            ax = axes[i, j]
                        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=10)
                        ax.axis('off')
        
        plt.suptitle('Sample Fingerprint Images by Blood Group', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_architecture(self, model, save_path='model_architecture.png'):
        """Plot model architecture summary"""
        # Create a simple representation of the model layers
        layers = []
        layer_types = []
        
        for layer in model.layers:
            layer_types.append(layer.__class__.__name__)
            if hasattr(layer, 'filters'):
                layers.append(f"{layer.__class__.__name__}\n{layer.filters} filters")
            elif hasattr(layer, 'units'):
                layers.append(f"{layer.__class__.__name__}\n{layer.units} units")
            else:
                layers.append(layer.__class__.__name__)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(layers))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        bars = ax.barh(y_pos, [1]*len(layers), color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers, fontsize=10)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_title('Model Architecture', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        
        # Add layer type annotations
        for i, (bar, layer_type) in enumerate(zip(bars, layer_types)):
            ax.text(0.5, bar.get_y() + bar.get_height()/2, layer_type,
                   ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_learning_curves(self, history, save_path='learning_curves.png'):
        """Plot detailed learning curves with smoothing"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Training Loss
        ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation Loss
        ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Validation Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training Accuracy
        ax3.plot(epochs, history.history['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
        ax3.set_title('Training Accuracy', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax4.plot(epochs, history.history['val_accuracy'], 'm-', label='Validation Accuracy', linewidth=2)
        ax4.set_title('Validation Accuracy', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_summary_report(self, history, model, save_path='summary_report.txt'):
        """Create a comprehensive summary report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("FINGERPRINT BLOOD GROUP PREDICTION - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 20 + "\n")
            # Capture model summary as string instead of writing directly
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            for line in summary_lines:
                f.write(line + '\n')
            
            f.write("\nTRAINING RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
            
            f.write("\nDATASET INFORMATION:\n")
            f.write("-" * 20 + "\n")
            for blood_group in self.blood_groups:
                group_path = os.path.join(self.dataset_path, blood_group)
                if os.path.exists(group_path):
                    count = len([f for f in os.listdir(group_path) if f.endswith('.BMP')])
                    f.write(f"{blood_group}: {count} samples\n")
        
        print(f"Summary report saved to {save_path}")

# Example usage function
def run_visualizations():
    """Run all visualizations for the project"""
    visualizer = BloodGroupVisualizer()
    
    print("Creating visualizations...")
    
    # Plot data distribution
    print("1. Plotting data distribution...")
    visualizer.plot_data_distribution()
    
    # Display sample images
    print("2. Displaying sample images...")
    visualizer.display_sample_images()
    
    print("Visualizations completed! Check the generated PNG files.")

if __name__ == "__main__":
    run_visualizations() 