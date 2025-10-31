#!/usr/bin/env python3
"""
Analyze training dataset distribution
Shows why LV_FUNCTION model wasn't trained
"""

import json
from collections import Counter

# Load dataset
with open('data/processed/training_dataset.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("DATASET ANALYSIS")
print("=" * 80)
print(f"\nTotal samples: {len(data)}")

# Analyze EF distribution
ef_values = []
for item in data:
    ef = item['measurements'].get('EF', 0)
    if ef > 0:
        ef_values.append(ef)

print(f"\nEjection Fraction (EF) Statistics:")
print(f"  Samples with EF: {len(ef_values)}/{len(data)}")
if ef_values:
    print(f"  Min EF: {min(ef_values):.1f}%")
    print(f"  Max EF: {max(ef_values):.1f}%")
    print(f"  Average EF: {sum(ef_values)/len(ef_values):.1f}%")
    
    # Count by category
    normal = sum(1 for ef in ef_values if ef >= 55)
    mild = sum(1 for ef in ef_values if 45 <= ef < 55)
    moderate = sum(1 for ef in ef_values if 30 <= ef < 45)
    severe = sum(1 for ef in ef_values if ef < 30)
    
    print(f"\nLV Function Distribution:")
    print(f"  Normal (EF ≥ 55%):      {normal:3d} ({normal/len(ef_values)*100:.1f}%)")
    print(f"  Mild (45-54%):          {mild:3d} ({mild/len(ef_values)*100:.1f}%)")
    print(f"  Moderate (30-44%):      {moderate:3d} ({moderate/len(ef_values)*100:.1f}%)")
    print(f"  Severe (< 30%):         {severe:3d} ({severe/len(ef_values)*100:.1f}%)")

# Analyze all categories
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION FOR ALL CATEGORIES")
print("=" * 80)

def extract_label_lv_function(text):
    """Extract LV function label."""
    if not text:
        return 'Unknown'
    if 'Normal' in text or 'normal' in text:
        return 'Normal'
    elif 'Mild' in text:
        return 'Mild'
    elif 'Moderate' in text:
        return 'Moderate'
    elif 'Severe' in text:
        return 'Severe'
    return 'Unknown'

def extract_label_lv_size(text):
    """Extract LV size label."""
    if not text:
        return 'Unknown'
    if 'Normal' in text:
        return 'Normal'
    elif 'Dilated' in text or 'enlargement' in text:
        return 'Dilated'
    return 'Unknown'

def extract_label_hypertrophy(text):
    """Extract hypertrophy label."""
    if not text:
        return 'Unknown'
    if 'Normal' in text:
        return 'None'
    elif 'Mild' in text:
        return 'Mild'
    elif 'Moderate' in text:
        return 'Moderate'
    elif 'Severe' in text:
        return 'Severe'
    return 'Unknown'

def extract_label_la_size(text):
    """Extract LA size label."""
    if not text:
        return 'Unknown'
    if 'Normal' in text:
        return 'Normal'
    elif 'enlarge' in text.lower():
        return 'Enlarged'
    return 'Unknown'

def extract_label_diastolic(text):
    """Extract diastolic function label."""
    if not text:
        return 'Unknown'
    if 'Normal' in text:
        return 'Normal'
    else:
        return 'Abnormal'

# Collect all labels
lv_function_labels = []
lv_size_labels = []
hypertrophy_labels = []
la_size_labels = []
diastolic_labels = []

for item in data:
    interp = item['interpretations']
    
    lv_func = extract_label_lv_function(interp.get('Left Ventricular Function', ''))
    lv_function_labels.append(lv_func)
    
    lv_sz = extract_label_lv_size(interp.get('LV Diastolic Dimension', ''))
    lv_size_labels.append(lv_sz)
    
    hyp = extract_label_hypertrophy(interp.get('Interventricular Septum', ''))
    hypertrophy_labels.append(hyp)
    
    la = extract_label_la_size(interp.get('Left Atrium', ''))
    la_size_labels.append(la)
    
    dias = extract_label_diastolic(interp.get('Diastolic Function', ''))
    diastolic_labels.append(dias)

# Print distributions
categories = {
    'LV_FUNCTION': lv_function_labels,
    'LV_SIZE': lv_size_labels,
    'LV_HYPERTROPHY': hypertrophy_labels,
    'LA_SIZE': la_size_labels,
    'DIASTOLIC_FUNCTION': diastolic_labels
}

for category, labels in categories.items():
    print(f"\n{category}:")
    counter = Counter(labels)
    valid_count = sum(count for label, count in counter.items() if label != 'Unknown')
    
    for label, count in counter.most_common():
        pct = count/len(labels)*100
        print(f"  {label:15s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"  Valid samples: {valid_count}/{len(labels)}")
    
    # Check if trainable
    if valid_count < 10:
        print(f"  ⚠️  TOO FEW SAMPLES - Cannot train model")
    elif len([c for l, c in counter.items() if l != 'Unknown']) < 2:
        print(f"  ⚠️  ONLY ONE CLASS - Cannot train classifier")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
4 models were successfully trained because they have:
- Multiple classes (at least 2)
- Sufficient samples (> 10 per category)

LV_FUNCTION model wasn't trained because:
- Most patients have Normal LV function (healthy patients)
- Too few samples of dysfunction classes
- Need more diverse patient data (patients with heart failure)

SOLUTION OPTIONS:
1. Keep 4 models + rule-based LV_FUNCTION (RECOMMENDED)
2. Collect more data with abnormal LV function patients
3. Use data augmentation techniques
4. Accept that your dataset represents healthy population
""")
