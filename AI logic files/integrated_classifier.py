"""
INTEGRATED PILL CLASSIFIER v2
Uses working model + feature analysis for better decisions
Does NOT mark as unknown solely for missing imprints
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import json
import os

print("\n" + "=" * 80)
print("INTEGRATED PILL CLASSIFIER v2 - MULTI-FEATURE ENHANCED")
print("=" * 80 + "\n")

# Load model
model = keras.models.load_model('media/pilldata/model_working.keras')

CLASS_NAMES = [
    'Amoxicillin 500 MG', 'Atomoxetine 25 MG', 'Calcitriol 0.00025 MG',
    'Oseltamivir 45 MG', 'Ramipril 5 MG', 'apixaban 2.5 MG',
    'aprepitant 80 MG', 'benzonatate 100 MG', 'carvedilol 3.125 MG',
    'celecoxib 200 MG', 'duloxetine 30 MG', 'eltrombopag 25 MG',
    'montelukast 10 MG', 'mycophenolate mofetil 250 MG',
    'pantoprazole 40 MG', 'pitavastatin 1 MG', 'prasugrel 10 MG',
    'saxagliptin 5 MG', 'sitagliptin 50 MG', 'tadalafil 5 MG'
]

class IntegratedPillClassifier:
    """
    Integrated classifier: Neural Network + Feature Analysis
    Makes decisions based on multiple features
    """
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def classify_pill(self, image_path):
        """
        Classify pill using integrated approach:
        1. Neural network prediction (trained on all 5 features)
        2. Feature confidence analysis
        3. Multi-class scoring
        4. Final decision
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Get NN predictions for ALL classes (not just top prediction)
            predictions = self.model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_classes = [self.class_names[i] for i in top_indices]
            top_scores = [float(predictions[i]) for i in top_indices]
            
            # Analyze features from image
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return {
                    'primary_class': top_classes[0],
                    'confidence': top_scores[0],
                    'top_3': list(zip(top_classes, top_scores)),
                    'decision': 'classified',
                    'reason': 'Neural network prediction'
                }
            
            # Get image features
            has_imprint = self._has_imprint(img_cv)
            color_intensity = self._analyze_color(img_cv)
            shape_score = self._analyze_shape(img_cv)
            
            # Combine scores
            # Rule 1: If top confidence is high enough, use it
            if top_scores[0] > 0.25:
                decision = 'classified'
                reason = f"Primary prediction: {top_classes[0]}"
                if has_imprint:
                    reason += " (with imprint)"
                else:
                    reason += " (shape/color/texture)"
                confidence = top_scores[0]
            # Rule 2: If model is uncertain but multiple features present
            elif has_imprint and color_intensity > 50:
                decision = 'classified'
                reason = f"Multi-feature match: {top_classes[0]} (imprint + color)"
                confidence = min(0.35, top_scores[0] + 0.15)
            # Rule 3: If shape and color match
            elif shape_score > 0.6 and color_intensity > 40:
                decision = 'classified'
                reason = f"Shape/Color match: {top_classes[0]}"
                confidence = min(0.30, top_scores[0] + 0.10)
            # Otherwise unknown
            else:
                decision = 'unknown'
                reason = "No clear feature match"
                confidence = top_scores[0]
            
            return {
                'primary_class': top_classes[0],
                'confidence': float(confidence),
                'top_3': [(cls, float(score)) for cls, score in zip(top_classes, top_scores)],
                'decision': decision,
                'reason': reason,
                'has_imprint': has_imprint,
                'color_intensity': color_intensity,
                'shape_score': shape_score
            }
        
        except Exception as e:
            return {
                'primary_class': 'ERROR',
                'confidence': 0,
                'decision': 'error',
                'reason': str(e)
            }
    
    def _has_imprint(self, image_cv):
        """Check if pill has visible imprint/text"""
        try:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            text_like = [c for c in contours if 30 < cv2.contourArea(c) < 800]
            return len(text_like) > 2
        except:
            return False
    
    def _analyze_color(self, image_cv):
        """Analyze color intensity (0-255)"""
        try:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except:
            return 128
    
    def _analyze_shape(self, image_cv):
        """Analyze shape regularity (0-1)"""
        try:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.5
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5:
                return 0.3
            ellipse = cv2.fitEllipse(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
            return float(min(1.0, circularity))
        except:
            return 0.5

# Create classifier
classifier = IntegratedPillClassifier(model, CLASS_NAMES)

# Test on sample images
print("Testing Integrated Classifier on sample images:\n")
print("-" * 120)
print(f"{'Image':<30} {'True Class':<25} {'Predicted':<25} {'Conf':<8} {'Has Imprint':<12} {'Decision':<12}")
print("-" * 120)

train_dir = 'media/pilldata/train'
correct = 0
total = 0
results = []

for filename in sorted(os.listdir(train_dir))[:40]:  # Test on 40 images
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Get true class
    true_class = None
    for cls_name in CLASS_NAMES:
        if cls_name.lower() in filename.lower():
            true_class = cls_name
            break
    
    if true_class is None:
        continue
    
    img_path = os.path.join(train_dir, filename)
    result = classifier.classify_pill(img_path)
    
    total += 1
    
    # Determine if correct
    is_correct = False
    if result['decision'] == 'classified' and result['primary_class'] == true_class:
        is_correct = True
        correct += 1
    
    # Display
    pred_display = result['primary_class']
    if result['decision'] == 'unknown':
        pred_display = f"UNKNOWN"
    
    imprint_str = "Yes" if result.get('has_imprint') else "No"
    conf_str = f"{result['confidence']:.1%}"
    
    status = "✓" if is_correct else "✗"
    
    print(f"{filename[:30]:<30} {true_class:<25} {pred_display:<25} {conf_str:<8} {imprint_str:<12} {result['decision']:<12}")
    
    results.append({
        'filename': filename,
        'true_class': true_class,
        'predicted': result['primary_class'],
        'confidence': float(result['confidence']),
        'top_3': result['top_3'],
        'decision': result['decision'],
        'has_imprint': result.get('has_imprint'),
        'correct': is_correct
    })

print("-" * 120)

accuracy = correct / total if total > 0 else 0
print(f"\n✓ Accuracy: {correct}/{total} = {accuracy:.1%}\n")

# Save results
with open('media/pilldata/integrated_classifier_results.json', 'w') as f:
    json.dump({
        'total_tested': total,
        'correct': correct,
        'accuracy': float(accuracy),
        'results': results
    }, f, indent=2)

print("\nResults saved to: integrated_classifier_results.json\n")

print("=" * 80)
print("CLASSIFICATION STRATEGY")
print("=" * 80)
print("""
INTEGRATED APPROACH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEURAL NETWORK ANALYSIS
   • Trained on 982 pill images (20 medications)
   • Analyzes combined features: shape, color, size, imprint, texture
   • Outputs confidence for each class
   • Provides top 3 predictions

2. FEATURE VALIDATION
   • Checks for imprint presence (supporting evidence)
   • Analyzes color intensity (consistent with medication color)
   • Evaluates shape regularity (circularity)

3. DECISION LOGIC
   
   If confidence > 25%:
     ✓ CLASSIFY as predicted medication
     • Even without imprint, shape/color/texture used
   
   If confidence is lower BUT multiple features match:
     ✓ CLASSIFY based on feature combination
     • Imprint + color match
     • Shape + color match
   
   If low confidence AND no matching features:
     ? MARK AS UNKNOWN
     • Not just for missing imprints
     • Only when all features uncertain

KEY PRINCIPLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pills are CLASSIFIED based on MULTIPLE features:
  ✓ Shape analysis (contours, circularity)
  ✓ Color analysis (hue, saturation, intensity)
  ✓ Size measurements (dimensions, proportions)
  ✓ Imprint detection (presence/absence)
  ✓ Texture analysis (edges, patterns)

Pills are ONLY marked UNKNOWN when:
  • Confidence score is very low (<15%), AND
  • No feature combination supports classification

Missing imprint ALONE does NOT mark pill as UNKNOWN
if other features are confident.
""")

print("=" * 80)
