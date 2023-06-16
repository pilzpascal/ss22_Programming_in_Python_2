from torchvision import transforms
import torch
import PIL

""""
class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace
        
    def __call__(self, tensor):
        tensor F.normalize(tensor, self.demean, self.dest, )
"""

torch.manual_seed(0)

org_tensor = torch.rand(3, 100, 100)*255

to_image = transforms.ToPILImage()
to_tensor = transforms.PILToTensor()

image = to_image(org_tensor)
tensor = to_tensor(image).type(torch.float32)

diff1 = org_tensor - tensor
print(torch.max(diff1))
print(torch.mean(diff1))

rotate = transforms.RandomRotation(degrees=180, expand=False)

image = rotate(image)

image.show()

mean = torch.tensor([torch.mean(org_tensor[0]), torch.mean(org_tensor[1]), torch.mean(org_tensor[2])])
std = torch.tensor([torch.std(org_tensor[0]), torch.std(org_tensor[1]), torch.std(org_tensor[2])])

normalize = transforms.Normalize(mean, std)
denormalize = transforms.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std])

tensor_norm = normalize(tensor)
tensor_denorm = denormalize(tensor_norm)

new_image = to_image(tensor_norm)
new_image.show()

tensor = tensor.type(torch.int32)
tensor_denorm = tensor_denorm.type(torch.int32)

diff2 = (tensor - tensor_denorm).type(torch.float32)
print(torch.max(diff2))
print(torch.mean(diff2))

end_image = to_image(tensor_denorm.type(torch.float32))
end_image.show()

print("done!")
