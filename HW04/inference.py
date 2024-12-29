import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from models.vgg_paper import VGGPaper
from dataset import get_transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FERPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = VGGPaper(num_classes=7)
        
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'Loaded checkpoint with accuracy: {checkpoint["accuracy"]:.2f}%')
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transforms(mode='test')
        
    def predict_image(self, image_path):
        """Predice la emoción para una imagen"""
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
            outputs = outputs.mean(0)
            probs = torch.softmax(outputs, dim=0)
            pred = outputs.argmax().item()
            confidence = probs[pred].item()
            
        return pred, confidence

def generate_submission(test_dir, checkpoint_path, output_path):
    """Genera archivo de submisión para Kaggle"""
    predictor = FERPredictor(checkpoint_path)
    
    # Lista de imágenes de test
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    results = []
    
    logger.info('Generating predictions...')
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_dir, img_name)
        try:
            pred, conf = predictor.predict_image(img_path)
            filename = os.path.splitext(img_name)[0]
            results.append({
                'filename': filename,  
                'label': pred,
                'confidence': conf
            })
        except Exception as e:
            logger.error(f'Error processing {img_name}: {str(e)}')
            filename = os.path.splitext(img_name)[0]  
            results.append({
                'filename': filename,
                'label': 0,
                'confidence': 0.0
            })
    
    df = pd.DataFrame(results)
    df = df[['filename', 'label']]  
    df.to_csv(output_path, index=False)
    logger.info(f'Submission saved to {output_path}')
    
    print("\nPrediction distribution:")
    print(df['label'].value_counts().sort_index())
    
    print("\nSample predictions (submission format):")
    print(df.head().to_string())

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for FER2013')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/Images/test',
                        help='Directory containing test images')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    args = parser.parse_args()
    
    generate_submission(args.test_dir, args.checkpoint, args.output)

if __name__ == '__main__':
    main()