import argparse
import os
import csv
import re
import torch
import time
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

CONFIDENCE_THRESHOLD = 0.01
MAX_IMAGE_SIDE = 1200

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
    local_25 = os.path.join(script_dir, "Qwen2.5-VL-7B-Instruct")
    local_7b = os.path.join(script_dir, "Qwen2-VL-7B-Instruct")

    if os.path.exists(local_25):
        model_path = local_25
    elif os.path.exists(local_7b):
        model_path = local_7b
    else:
        raise RuntimeError("Model not found locally. Did setup.bash run?")

    print(f"Loading model from: {model_path}")

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        max_memory = {i: "13GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "20GiB"
    elif n_gpus == 1:
        max_memory = {0: "13GiB", "cpu": "20GiB"}
    else:
        max_memory = None

    model = QwenVLModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def get_logprob_answer(inputs, model, processor):
    """CoT greedy pass with output_scores; confidence = full-vocab prob of the digit
    at the exact generation step where the model wrote 'Answer: X'."""
    digit_token_map = {}
    for d in range(1, 5):
        for surface in [str(d), " " + str(d)]:
            tids = processor.tokenizer.encode(surface, add_special_tokens=False)
            if len(tids) == 1:
                digit_token_map[tids[0]] = d

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_scores=True,
            return_dict_in_generate=True,
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = gen.sequences[0][input_len:].tolist()
    scores = gen.scores

    cot_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    answer = None
    confidence = 0.0
    for i, tid in enumerate(generated_ids):
        if tid in digit_token_map:
            prev_text = processor.tokenizer.decode(generated_ids[:i], skip_special_tokens=True)
            if re.search(r'Answer:\s*$', prev_text):
                answer = digit_token_map[tid]
                probs = F.softmax(scores[i][0], dim=0)
                confidence = probs[tid].item()
                break

    return cot_text, answer, confidence


def predict_answer(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        ratio = MAX_IMAGE_SIDE / max(image.size)
        image = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)),
            Image.LANCZOS,
        )

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

    # PRIMARY: model explicitly wrote "Answer: X" with sufficient confidence
    if answer is not None and confidence >= CONFIDENCE_THRESHOLD:
        return answer

    # FALLBACK: skip
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
    print("Loading Qwen2.5-VL-7B model...")
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

    with open(submission_path, "w", newline="") as f:
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
