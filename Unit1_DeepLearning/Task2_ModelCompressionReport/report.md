# Research Mini-Survey: Model Compression Techniques

## Unit 1, Task 2:

## Compare model compression methods: Pruning, Quantization, and Knowledge Distillation

### Author: Deepesh

---

## Introduction

Deep learning (DL) models are getting bigger and need more computing power. This makes it hard to use them on small devices like mobile phones, simple gadgets (edge devices) or in apps that need to run instantly (real-time applications).

Model compression techniques are ways to shrink these models. The goal is to use less memory, need less calculation time and run faster, all while trying to keep the model's performance (accuracy) high.

**This mini-survey explores three major techniques:**

### 1. Pruning

### 2. Quantization

### 3. Knowledge Distillation

---

## 1. Pruning

Pruning removes unnecessary weights, neurons, or attention heads from a trained model. The idea is that many parameters contribute very little to the final output and can be removed. Examples : MobileNetV2 (Image Classification), BERT (Large Language Models - LLMs), VGG-16 (Early CNN Architecture).

### Types of Pruning

- Weight pruning (remove low-magnitude weights)
- Neuron pruning
- Layer pruning
- Attention-head pruning (Transformers)

### Benefits

- Up to 50–90% reduction in parameters
- Faster inference
- Lower RAM usage

### Few Real-World Examples

- **Cloud Service Cost Reduction:**
  For massive cloud services like image search, pruning large backbone models by 20% - 30% results in millions of dollars saved annually. Lower computation demands translate directly to a need for fewer GPUs and reduced electricity consumption across global data centers.

- **Always-On Smart Speakers:**
  Unstructured Weight Pruning creates sparse matrices in acoustic models for trigger word detection. This allows smart speakers to operate in an ultra-low-power, always-on state with minimal battery drain and near-zero latency for recognizing the wake word, maintaining immediate responsiveness.

- **Autonomous Vehicles:**
  Pruning object detection models with Structured Filter Pruning reduces computational load by 30% - 50%. This increase in the inference frame rate (FPS) is critical for autonomous systems to guarantee instantaneous decision-making and vehicle safety in fast-changing road environments.

---

## 2. Quantization

Quantization reduces the precision of weights and activations from floating-point (FP32) to lower-bit formats such as FP16, INT8, or INT4. Examples :ResNet-50 (Image Classification), LLaMA Series (Large Language Models), SqueezeNet (Highly Compact CNN).

### Types

- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Dynamic quantization

### Benefits

- 4× reduction in model size (FP32 → INT8)
- Faster CPU inference
- Very small accuracy drop

### Few Real-World Examples

- **Mobile Face Recognition:**
  Quantization reduces a face detection model's size from 100MB to 25MB on an Android phone by converting weights to 8-bit integers. This allows the model to run offline and instantly when unlocking the phone or organizing photos, consuming far less battery power than running the original high-precision model.

- **Voice Assistants (Edge Devices):**
  Smart speakers utilize quantization for their "wake word" detection models. By simplifying the calculations, the model runs continuously on a low-power processor using minimal electricity, allowing the device to remain "always-on" and ready to respond immediately without needing to constantly access cloud servers.

- **Augmented Reality (AR) Filters:**
  Real-time AR filters, like those that track your hands or segment the background, are powered by quantized models. These models run 3 to 4 times faster on a mobile GPU's integer units, ensuring the virtual effects track perfectly without the noticeable lag that a high-precision model would cause.

---

## 3. Knowledge Distillation

### What is Distillation?

A smaller model (student) is trained to mimic a larger model (teacher) by learning from its soft predictions. Examples : DistilBERT (LLM Student), Image Recognition (Ensemble Compression), Speech Recognition (Acoustic Models).

### Benefits

- 2×–20× reduction in size
- Nearly same accuracy
- Works extremely well for NLP & Transformers

### Real-World Examples

- **Google Search Ranking Models:**
  Google trains a massive, highly accurate Teacher model to learn complex search ranking logic. Knowledge Distillation then transfers this logic to a much smaller Student model, allowing search results to be returned instantly and efficiently across billions of daily queries with minimal loss in ranking quality.

- **On-Device Language Translation:**
  A large cloud-based transformer model acts as the Teacher to teach a small, specialized mobile model (the Student) the nuances of language. This enables the student model to perform high-quality translation offline on a smartphone app, like Google Translate, without needing constant network access.

- **Speech Command Recognition:**
  A complex acoustic model (the Teacher) provides "soft labels"—its probability distribution for all potential words—to a tiny Student model designed for an edge microcontroller. The student uses this distilled knowledge to accurately recognize a limited set of voice commands with very low latency and power consumption.

---

## Comparison of all three

| Technique    | Size Reduction | Speedup | Accuracy Loss | Best Use Case      |
| ------------ | -------------- | ------- | ------------- | ------------------ |
| Pruning      | Medium         | Medium  | Low–Medium    | CNNs, Transformers |
| Quantization | High           | High    | Very Low      | Mobile/Edge AI     |
| Distillation | Very High      | Medium  | Very Low      | Large NLP models   |

---

## Conclusion

Pruning, quantization and knowledge distillation are powerful tools that make deep learning models efficient and deployable on resource-constrained devices. Quantization gives the best speedup, pruning reduces unnecessary structure and distillation provides compact models with minimal accuracy drop.

Together, these techniques enable real-world deployment of modern deep learning systems.

---

# End of Report
