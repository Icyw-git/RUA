# Save huggingface checkpoint as simplified models
import os
import sys
sys.path.append(os.getcwd())
from models.wla import WLA

model_path = "your_model_path"
wla = WLA.from_pretrained(
    model_path,
)
wla.save_pretrained("save_dir")