import torch
from PIL import Image
from torchvision import transforms

def pred_image(image_path, model, device):
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([
       transforms.Resize(226),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)])
   
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output = model(img_normalized)

      # Reverse the log function in our output
      output = torch.exp(output)
      
      # Get the top predicted class, and the output percentage for
      # that class
      probs, classes = output.topk(1, dim=1)
      label = ['busuk', 'normal-busuk', 'sangat-segar', 'segar-normal']
      acc = "%.1f" % (probs.item() * 100)
      cls = label[classes.item()]
      return [cls, acc]

    #   index = output.data.cpu().numpy().argmax()
    #   class_name = classes[index]
    #   return class_name