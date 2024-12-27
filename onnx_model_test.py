import onnxruntime as ort
from tqdm import tqdm
import glob
import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_image(image, target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_normalized = (image_normalized - mean) / std
    image_transposed = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    return image_transposed

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    softmax_output = exp_logits / np.sum(exp_logits)
    return softmax_output

def load_and_preprocess(img_path):
    data = cv2.imread(img_path)
    label = 1
    data = preprocess_image(data)
    return img_path, data, label

def main(model_path, dir_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_sess = ort.InferenceSession(model_path)
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[1].name
    results = []

    output_csv_path = './swin_transformer_real_13k_onnx.csv'

    img_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
    batch_size = 4

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in tqdm(range(0, len(img_paths), batch_size)):
            batch_paths = img_paths[i:i + batch_size]

            for img_path in batch_paths:
                futures.append(executor.submit(load_and_preprocess, img_path))

            batch_data = []
            batch_labels = []
            batch_paths_loaded = []

            for future in as_completed(futures):
                img_path, data, label = future.result()
                batch_data.append(data)
                batch_labels.append(label)
                batch_paths_loaded.append(img_path)

            batch_data = np.array(batch_data)
            results_batch = ort_sess.run([output_name], {input_name: batch_data})

            for j, result in enumerate(results_batch[0]):
                prob = softmax(result)
                class_1_prob, class_2_prob = prob
                results.append([batch_paths_loaded[j], batch_labels[j], f"{class_2_prob:.10f}"])

            df = pd.DataFrame(results, columns=["Image Name", "label", "spoof"])
            df.to_csv(output_csv_path, index=False)

def test_cam(model_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_sess = ort.InferenceSession(model_path)
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[1].name
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    while True:
        success, img = cap.read()
        if not success:
            break

        data = preprocess_image(img)
        results = ort_sess.run([output_name], {input_name: data})
        prob = softmax(results[0])
        class_1_prob, class_2_prob = prob
        if class_2_prob>=0.8:
            img_is="spoof"
        else:
            img_is = "real"

if __name__ == '__main__':
    model_path = "swinV2_base.onnx"
    dir_path = r"C:\Users\Mantra\Documents\facial\liveness_models\real_13k_face_f3"
    # main(model_path, dir_path)
    test_cam(model_path)