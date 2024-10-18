import onnxruntime as ort
from tqdm import tqdm
import glob
import os
import cv2
import numpy as np
import pandas as pd


def preprocess_image(image, target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Resize the image to the target size
    image_resized = cv2.resize(image, target_size)

    # Convert from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Convert pixel values to [0, 1] by dividing by 255
    image_normalized = image_rgb / 255.0

    # Normalize the image using mean and std
    image_normalized = (image_normalized - mean) / std

    # Transpose to convert the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format
    image_transposed = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    # image_transposed = np.expand_dims(image_transposed, axis=0)
    # image_tensor = torch.tensor(image_transposed)
    return image_transposed


def softmax(logits):
    # Apply softmax function manually
    exp_logits = np.exp(logits - np.max(logits))
    softmax_output = exp_logits / np.sum(exp_logits)
    return softmax_output


def main(model_path, dir_path):
    ort_sess = ort.InferenceSession(model_path)
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[1].name
    results = []
    output_csv_path = './swin_transformer_spoof_59k_onnx.csv'

    img_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
    batch_size=32
    # Process images in batches
    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[i:i + batch_size]
        batch_data = []
        batch_labels = []

        # Load and preprocess images
        for img_path in batch_paths:
            data = cv2.imread(img_path)
            # if "spoof" in img_path:
            #     label = 1
            # else:
            label = 1
            data = preprocess_image(data)
            batch_data.append(data)
            batch_labels.append(label)

        # Convert batch to a single numpy array (assuming model expects a batch of images)
        batch_data = np.array(batch_data)

        # Run the model on the batch
        results_batch = ort_sess.run([output_name], {input_name: batch_data})

        # Post-process the results
        for j, result in enumerate(results_batch[0]):
            prob = softmax(result)
            # print(prob)
            class_1_prob, class_2_prob = prob
            # Append the results
            results.append([batch_paths[j], batch_labels[j], f"{class_2_prob:.10f}"])
        df = pd.DataFrame(results, columns=["Image Name", "label", "spoof"])
        # Write the DataFrame to a CSV file
        df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    model_path = "swinV2_base.onnx"
    dir_path = r"C:\Users\Mantra\Documents\facial\liveness_models\face_folder0"
    main(model_path, dir_path)
