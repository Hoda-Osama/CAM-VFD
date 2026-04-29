#!/bin/bash

# Run robustness evaluations for both CogVideo and SVD subsets
echo "Starting Robustness Evaluations..."

# Evaluate on CogVideo subset
python evaluate_robustness.py --subset cogvideo --evaluation_type all --model_path best_model.pth

# Evaluate on SVD subset
python evaluate_robustness.py --subset svd --evaluation_type all --model_path best_model.pth

# Evaluate only signal degradation
python evaluate_robustness.py --subset both --evaluation_type signal --model_path best_model.pth

# Evaluate only photometric perturbations
python evaluate_robustness.py --subset both --evaluation_type photometric --model_path best_model.pth

# Evaluate only adversarial attacks
python evaluate_robustness.py --subset both --evaluation_type adversarial --model_path best_model.pth

echo "Robustness evaluations completed!"