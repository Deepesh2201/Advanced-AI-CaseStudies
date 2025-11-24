# Research Mini-Survey: Model Compression Techniques

## Unit 1, Task 2:

## Compare model compression methods: Pruning, Quantization, and Knowledge Distillation

### Author: Deepesh

---

## 1. Introduction

Deep learning (DL) models are getting bigger and need more computing power. This makes it hard to use them on small devices like mobile phones, simple gadgets (edge devices) or in apps that need to run instantly (real-time applications).

Model compression techniques are ways to shrink these models. The goal is to use less memory, need less calculation time and run faster, all while trying to keep the model's performance (accuracy) high.

**This mini-survey explores three major techniques:**

- ## Pruning
- ## Quantization
- ## Knowledge Distillation

---

## 2. Pruning

### What is Pruning?

Pruning removes unnecessary weights, neurons, or attention heads from a trained model. The idea is that many parameters contribute very little to the final output and can be removed.

### Types of Pruning

- Weight pruning (remove low-magnitude weights)
- Neuron pruning
- Layer pruning
- Attention-head pruning (Transformers)

### Benefits

- Up to 50–90% reduction in parameters
- Faster inference
- Lower RAM usage

### Real-World Examples

- **Google MobileNet-V2 pruning** – MobileNetV2 achieved a $\approx 30\% \text{ to } 40\%$ inference speedup over MobileNetV1 primarily through its novel architecture, specifically the inverted residual blocks and linear bottlenecks. To further optimize MobileNetV2 for deployment, researchers apply Structured Channel Pruning, which involves removing entire low-importance filters in the convolutional layers. This process significantly reduces the model's computational load (FLOPs) and parameters—often achieving an additional $50\%$ reduction in FLOPs—while employing fine-tuning to ensure the accuracy drop remains minimal, making the model faster and smaller for mobile and edge devices.

- OpenAI GPT-2 pruning – reduces parameter count without major accuracy loss
- TensorFlow Model Optimization Toolkit offers magnitude-based pruning for edge devices

---

## 3. Quantization

### What is Quantization?

Quantization reduces the precision of weights and activations from floating-point (FP32) to lower-bit formats such as FP16, INT8, or INT4.

### Types

- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Dynamic quantization

### Benefits

- 4× reduction in model size (FP32 → INT8)
- Faster CPU inference
- Very small accuracy drop

### Real-World Examples

- LLaMA-3 4-bit quantization# enables on-device LLM inference
- Google TensorFlow Lite 8-bit quantization# for Android apps
- OpenAI Whisper quantized# runs fast on laptops with small accuracy loss

---

## 4. Knowledge Distillation

### What is Distillation?

A smaller model (student) is trained to mimic a larger model (teacher) by learning from its soft predictions.

### Benefits

- 2×–20× reduction in size
- Nearly same accuracy
- Works extremely well for NLP & Transformers

### Real-World Examples

- DistilBERT is 40% smaller than BERT but retains 97% accuracy
- TinyBERT designed for mobile NLP tasks
- MobileNet distilled for real-time detection on phones

---

## 5. Summary Comparison

| Technique    | Size Reduction | Speedup | Accuracy Loss | Best Use Case      |
| ------------ | -------------- | ------- | ------------- | ------------------ |
| Pruning#     | Medium         | Medium  | Low–Medium    | CNNs, Transformers |
| Quantization | High           | High    | Very Low      | Mobile/Edge AI     |
| Distillation | Very High      | Medium  | Very Low      | Large NLP models   |

---

## 6. Conclusion

Pruning, quantization, and knowledge distillation are powerful tools that make deep learning models efficient and deployable on resource-constrained devices. Quantization gives the best speedup, pruning reduces unnecessary structure, and distillation provides compact models with minimal accuracy drop.

Together, these techniques enable real-world deployment of modern deep learning systems.

---

# End of Report
