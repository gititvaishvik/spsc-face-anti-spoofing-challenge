import os
import glob
from tqdm import tqdm
from nets.swin_transformer_v2 import swin_v2_b
from nets.swin_transformer_v2 import load_pretrain, preprocess_image
import torch.nn.functional as F
import cv2
import pandas as pd
import torch
if __name__ == '__main__':
    model = swin_v2_b(num_classes=2, fp16=False)

    # weight_path = r"C:\Users\Mantra\Downloads\face_swin_v2_base (1).pth"
    weight_path = "weights/face_swin_v2_base (1).pth"
    model = load_pretrain(model, pretrain=weight_path)
    # model.eval()
    print(model)
    torch.save(model.state_dict(), 'swinV2_base_without_extra_layers.pth')
    # raise "stop here"
    # model.half()

    dir_path = r"C:\Users\Mantra\Documents\facial\liveness_models\spppf_test_images"
    results = []
    output_csv_path = './swin_transformer_scores_v2_2.csv'
    for img_path in tqdm(glob.glob(os.path.join(dir_path, "**", "*.jpg"))):
        data = cv2.imread(img_path)
        if "spoof" in img_path:
            label = 1
        else:
            label = 0
        data = preprocess_image(data)
        # print(data.shape)
        data = data.cpu().half()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(data)
        #
        # logit = out.view(1, n, 2)
        # logit = torch.mean(out[1], dim=0, keepdim=False)
                prob = F.softmax(out[1], 1)
        # Get the two class probabilities
        class_1_prob, class_2_prob = prob[0].cpu().numpy()

        # Append the results to the list
        results.append([img_path, label, f"{class_1_prob:.10f}", f"{class_2_prob:.10f}"])

        # print(out[0].shape)

    df = pd.DataFrame(results, columns=["Image Name", "label", "real", "spoof"])
    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)