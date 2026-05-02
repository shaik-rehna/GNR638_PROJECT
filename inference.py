import argparse
import os
import csv
import re
import torch
import time
import random
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CONFIDENCE_THRESHOLD = 0.01

SYSTEM_PROMPT = """You are an expert in deep learning, CNNs, and machine learning with PhD-level knowledge.

Key concepts you know well:
- CNN: convolution, pooling, stride, padding, receptive field, feature maps
- Backpropagation, vanishing/exploding gradients, chain rule
- Optimizers: SGD, Adam, RMSprop, momentum, learning rate scheduling
- Activation functions: ReLU, sigmoid, tanh, softmax, GELU
- Regularization: dropout, batch normalization, layer normalization, weight decay
- Loss functions: cross-entropy, MSE, focal loss
- Architectures: ResNet, VGG, Inception, EfficientNet, ViT, BERT, transformers
- Training: overfitting, underfitting, bias-variance tradeoff, data augmentation

Here are examples of how to reason:

Example 1:
Q: What is the output size of a 32x32 image after a Conv2D with 64 filters, 3x3 kernel, stride=1, padding=0?
Reasoning: Output = (input - kernel + 2*padding) / stride + 1 = (32 - 3 + 0) / 1 + 1 = 30. So output is 30x30x64.
Answer: 2

Example 2:
Q: Which of the following helps prevent vanishing gradients in deep networks?
A. Sigmoid activation  B. ReLU activation  C. Small learning rate  D. No skip connections
Reasoning: Sigmoid squashes gradients to near zero in deep networks. ReLU has gradient=1 for positive values, so it does not vanish. Skip connections (ResNet) also help but option B is the direct answer.
Answer: 2

Example 3:
Q: In batch normalization, what is normalized?
A. Weights  B. Gradients  C. Activations  D. Loss
Reasoning: Batch norm normalizes the activations (outputs) of each layer across the batch to have zero mean and unit variance.
Answer: 3

Now look at the MCQ image carefully and answer the question shown.

Step 1: Read the question and all 4 options carefully.
Step 2: Apply your deep learning knowledge to reason through each option.
Step 3: Eliminate wrong options and select the best answer.

End your response with exactly: "Answer: X" where X is 1, 2, 3, or 4.
1 = Option A
2 = Option B
3 = Option C
4 = Option D

If the image is unreadable or the question makes no sense, do NOT write "Answer:" at all."""

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_7b = os.path.join(script_dir, "Qwen2-VL-7B-Instruct")

    if not os.path.exists(local_7b):
        raise RuntimeError("Model not found locally. Did setup.bash run?")

    model_path = local_7b
    print(f"Loading model from: {model_path}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)

    return model, processor



def get_logprob_answer(inputs, model, processor):
    """Chain-of-thought greedy pass, then check digit probability against FULL vocabulary."""
    digit_token_ids = [
        processor.tokenizer.encode(str(d), add_special_tokens=False)[0]
        for d in range(1, 5)
    ]

    with torch.no_grad():
        cot_output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    cot_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, cot_output)
    ]
    cot_text = processor.batch_decode(cot_trimmed, skip_special_tokens=True)[0].strip()

   
    answer_suffix = processor.tokenizer.encode("\nAnswer:", add_special_tokens=False, return_tensors="pt")[0].to(inputs.input_ids.device)
    full_ids = torch.cat([cot_output[0], answer_suffix]).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits[0, -1]  # shape: [vocab_size]

    full_probs = F.softmax(logits, dim=0)
    digit_probs = [full_probs[tid].item() for tid in digit_token_ids]

    best_idx = int(torch.tensor(digit_probs).argmax().item())
    best_prob = digit_probs[best_idx]
    best_digit = best_idx + 1

    return cot_text, best_digit, best_prob

def extract_answer_from_cot(cot_text):
    match = re.search(r"Answer:\s*([1-4])", cot_text)
    if match:
        return int(match.group(1))
    match = re.search(r"Answer:\s*([A-D])", cot_text, re.IGNORECASE)
    if match:
        return "ABCD".index(match.group(1).upper()) + 1
    return None

def predict_answer(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    cot_text, answer, confidence = get_logprob_answer(inputs, model, processor)

    # PRIMARY: use what the model explicitly wrote
    cot_answer = extract_answer_from_cot(cot_text)
    if cot_answer is not None:
        return cot_answer

    # FALLBACK: logprob signal
    if confidence >= CONFIDENCE_THRESHOLD:
        return answer

    return 5  
    

def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    args = parser.parse_args()

    test_dir = args.test_dir
    test_csv_path = os.path.join(test_dir, "test.csv")
    images_dir = os.path.join(test_dir, "images")

    # Read test.csv
    image_names = []
    with open(test_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_names.append(row["image_name"])

    print(f"Found {len(image_names)} images to process")

    # Load model
    print("Loading Qwen2-VL-7B model...")
    model, processor = load_model()
    print("Model loaded successfully")

    # Run inference
    results = []
    total_start = time.time()

    for idx, image_name in enumerate(image_names):

        if image_name.endswith(".png"):
            image_file = image_name
        else:
            image_file = image_name + ".png"

        image_path = os.path.join(images_dir, image_file)

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping (marking as 5)")
            results.append({"id": image_name, 
                            "image_name": image_name, 
                            "option": 5})
            continue

        img_start = time.time()
        print(f"Processing {image_name} ({idx+1}/{len(image_names)})...")
        try:
            answer = predict_answer(image_path, model, processor)
        except Exception as e:
            print(f"Error on {image_name}: {e}")
            answer = 1  # safe fallback
        elapsed = time.time() - img_start
        total_elapsed = time.time() - total_start
        remaining = (total_elapsed / (idx + 1)) * (len(image_names) - idx - 1)
        print(f"  -> Answer: {answer} | {elapsed:.1f}s | ETA: {remaining/60:.1f}min")
        results.append({
            "id": image_name,
            "image_name": image_name,
            "option": answer
        })

    # Write submission.csv in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_path = os.path.join(script_dir, "submission.csv")

    with open("submission.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_name", "option"])
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - total_start
    print(f"\nSubmission saved to {submission_path}")
    print(f"Total time: {total_time/60:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    start = time.time()

    main()

    end = time.time()
    print(f"\nTotal execution time: {(end - start)/60:.2f} minutes")
