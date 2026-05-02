# GNR638 Project 2 - Visual MCQ Solver

**Team Members:**
- 21D180037
- 22B3932
- 23B4207

## Approach
We use **Qwen2.5-VL-7B-Instruct**, a Vision Language Model (VLM) that can read and reason about images containing text. The model reads each MCQ image, applies chain-of-thought reasoning with deep learning domain knowledge, and outputs the correct answer (1-4) or 5 to skip if uncertain.

## Environment Setup
Run the setup script (internet required):
```bash
bash setup.bash
conda activate gnr_project_env
```

## Inference
```bash
python inference.py --test_dir <absolute_path_to_test_dir>
```
This will generate `submission.csv` in the same directory as `inference.py`.

## Requirements
- Python 3.11
- CUDA-compatible GPU (tested on L40s 48GB)
- See `requirements.txt` for Python dependencies

## Files
- `inference.py` - Main inference script
- `setup.bash` - Environment setup and model download
- `requirements.txt` - Python dependencies
