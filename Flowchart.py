import EncoderGenerator
import torch

netE = EncoderGenerator.Encoder()
netG = EncoderGenerator.Generator()

input_names = ["Iris"]
output_names = ["Iris Species Prediction"]

torch.onnx.export(netE , X, "netE.onnx", input_names=input_names, output_names=output_names)