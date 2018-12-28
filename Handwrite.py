import torch
import torchvision
from torchvision import transforms, datasets
import time
import model
import dataset

model = model.make_model()
model.load_state_dict(torch.load('./models/LeNet_best_weights.pkl'))
print(model)

transform = transforms.Compose([
    transforms.Resize(size=34),
    transforms.RandomCrop(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image = datasets.ImageFolder(root='./handwrite', transform=transform)
images = torch.utils.data.DataLoader(dataset=image, batch_size=5, shuffle=True)

for data in images:
    since = time.time()
    inputs, labels = data
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    pred = predicted.numpy()
    print('predict labels is\n {}'.format(pred))
    out = torchvision.utils.make_grid(inputs)  # put a batch of images together
    dataset.imshow(out, title=[pred[x] for x in range(5)])
    batch_time = time.time() - since
    print('Classify a batch of images complete in %.2f s' % batch_time)
