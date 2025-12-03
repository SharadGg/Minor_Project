"""Predictor Class for Inference"""
import torch
import numpy as np

class GesturePredictor:
    """Simplified predictor for batch inference."""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, keypoints):
        """Predict on keypoints sequence."""
        with torch.no_grad():
            tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).item()
            conf = torch.softmax(output, dim=1).max().item()
        return pred, conf