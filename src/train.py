from clearml import Task, Dataset, OutputModel
import os
from ultralytics import YOLO


# Train YOLO v8 via ClearML

# 1. Init ClearML task
task = Task.init(project_name="fin-analyser", task_name="yolo")
task.execute_remotely(queue_name="gpu", clone=True)
# 2. Get the versioned dataset (no re-download if already local)
dataset = Dataset.get(dataset_id="d0a71c53f6834a6cb5761d0894936aa6")
data_path = dataset.get_local_copy()
yaml_path = os.path.join(data_path, "data.yaml")

# 3. Select model and hyperparams
model_variant = "yolov8n"
task.set_parameter("model_variant", model_variant)
model = YOLO(f"{model_variant}.pt")

# 4. Define args (note the "name" so checkpoints go to runs/train/fin_run)
args = {"data": yaml_path, "device": 0, "batch": 16, "epochs": 200, "name": "fin_run_1"}
task.connect(args)

# 5. Train
results = model.train(**args)

# 6. Upload artifacts
best_pt = os.path.join("runs", "train", args["name"], "weights", "best.pt")
last_pt = os.path.join("runs", "train", args["name"], "weights", "last.pt")
task.upload_artifact(name="yolo_best", artifact_object=best_pt)
task.upload_artifact(name="yolo_last", artifact_object=last_pt)

output_model = OutputModel(
    task=task,
    framework="PyTorch",
    name="yolov8-fin-v1",
    comment="YOLOv8 fine-tuned",
    config_dict=args,
)
# now attach the actual weights
output_model.update_weights(best_pt)
