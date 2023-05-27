import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd


DATA_DIR = Path('/hdd/bdhammel/photo_dataset/')


class Dataset(ImageFolder):

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        path = Path(self.imgs[index][0]).relative_to(self.root)
        return img, str(path)


transform = timm.data.create_transform((3, 224, 224))
dataset = Dataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = timm.create_model('eva02_base_patch16_clip_224', pretrained=True)

model.cuda()
vectors = []

for x, path in tqdm(dataloader):
    x = x.cuda()
    o = model(x)

    o = o.detach().cpu().numpy()

    # save vectors
    for emb, p in zip(o, path):
        vectors.append((p, emb))


df = pd.DataFrame(vectors, columns=['path', 'vector'])
df.to_pickle(DATA_DIR/'db.pkl')
