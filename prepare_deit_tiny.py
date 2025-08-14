# from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
# from PIL import Image
# import requests
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
# model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
# inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])


import torch

# 从facebookresearch/deit的GitHub主分支加载模型
# 'deit_tiny_distilled_patch16_224' 是蒸馏版DeiT-Tiny模型的官方名称
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)

# 将模型设置为评估模式
model.eval()

# 示例：创建一个随机的224x224图像张量
dummy_input = torch.randn(1, 3, 224, 224)

# 进行推理
with torch.no_grad():
    output = model(dummy_input)

# 输出的output是模型的logits
print(model)
model = model.to('cuda')