import onnx
import cv2
from nets.swin_transformer_v2 import swin_v2_b
from nets.swin_transformer_v2 import load_pretrain, preprocess_image
import torch


def convert(torch_model, torch_input, out_onnx_path):
    try:
        # Ensure the model is in evaluation mode
        torch_model.eval()

        # Export the model to ONNX
        # onnx_program = torch.onnx.dynamo_export(torch_model, torch_input, opset_version=11)
        torch.onnx.export(
            torch_model,  # model being run
            torch_input,  # model input (or a tuple for multiple inputs)
            out_onnx_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=False,  # whether to execute constant folding for optimization
            input_names=['input'],  # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                          'output': {0: 'batch_size'}}
        )

        print("Model exported successfully!")

        # print("Model exported successfully!")
    except Exception as e:
        print(f"Failed to export the model to ONNX: {e}")


def main():
    model = swin_v2_b(num_classes=2, fp16=False)
    # print(model)
    onnx_model_path = "./swinV2_base.onnx"
    weight_path = r"C:\Users\Mantra\Downloads\face_swin_v2_base (1).pth"
    model = load_pretrain(model, pretrain=weight_path)
    # Convert model to FP16
    model.half()

    # Save the FP16 weights
    torch.save(model.state_dict(), 'swinV2_base_model_fp16_weights.pth')
    # img = cv2.imread(r"C:\Users\Mantra\Documents\facial\liveness_models\spppf_test_images\live\000074_face_1.jpg")
    # data = preprocess_image(img)
    # convert(model, data,onnx_model_path)
    # model_onnx = onnx.load(onnx_model_path)
    # onnx.checker.check_model(model_onnx)


if __name__ == '__main__':
    main()
