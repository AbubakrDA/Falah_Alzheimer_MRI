
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import matplotlib.pyplot as plt

# Load the downloaded dataset
dataset_path = Path(__file__).resolve().parent if (Path(__file__).resolve().parent / "dataset_dict.json").exists() else Path("Falah_Alzheimer_MRI")
dataset = load_from_disk(str(dataset_path))

train_data = dataset["train"]
test_data = dataset["test"]

print("========== DATASET OVERVIEW ==========")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))

# Read class names from dataset metadata
label_feature = train_data.features["label"]

if hasattr(label_feature, "names") and label_feature.names:
    class_names = label_feature.names
else:
    class_names = [str(i) for i in range(len(set(train_data["label"])))]

print("\nClasses:", class_names)

# Class distribution
train_counts = Counter(train_data["label"])
test_counts = Counter(test_data["label"])

print("\n========== TRAINING DISTRIBUTION ==========")
for label, count in sorted(train_counts.items()):
    print(f"{class_names[label]}: {count}")

print("\n========== TEST DISTRIBUTION ==========")
for label, count in sorted(test_counts.items()):
    print(f"{class_names[label]}: {count}")

# Inspect image properties
sample = train_data[0]["image"]

print("\n========== IMAGE INFORMATION ==========")
print("Image size:", sample.size)
print("Image mode:", sample.mode)

# Display sample MRI images
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    example = train_data[i]
    ax.imshow(example["image"].convert("RGB"))
    ax.set_title(class_names[example["label"]])
    ax.axis("off")

plt.suptitle("NeuroVision - Sample Brain MRI Images")
plt.tight_layout()
plt.show()

# Plot training class distribution
plt.figure(figsize=(10, 5))

labels = [class_names[i] for i in sorted(train_counts)]
counts = [train_counts[i] for i in sorted(train_counts)]

plt.bar(labels, counts)
plt.title("Alzheimer MRI Training Dataset Distribution")
plt.xlabel("Disease Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
