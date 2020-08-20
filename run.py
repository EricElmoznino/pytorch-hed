from argparse import ArgumentParser
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as transforms
from PIL import Image
from HierarchicalEdgeDetector import HierarchicalEdgeDetector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ImageDataset(Dataset):
	def __init__(self, data_dir):
		super().__init__()
		self.data_dir = data_dir
		self.image_names = os.listdir(data_dir)

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, item):
		name = self.image_names[item]
		image = Image.open(os.path.join(self.data_dir, name)).convert('RGB')
		image = transforms.to_tensor(image)
		return image, name


if __name__ == '__main__':
	parser = ArgumentParser(description='Get image edge maps')
	parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
	parser.add_argument('--save_dir', required=True, type=str, help='path to save resulting edge maps')
	parser.add_argument('--saved_model_path', default='network-bsds500.pth', type=str, help='path of saved PyTorch model')
	args = parser.parse_args()

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	model = HierarchicalEdgeDetector(args.saved_model_path)
	dataloader = DataLoader(ImageDataset(args.data_dir), batch_size=64, num_workers=2)

	for images, image_names in dataloader:
		with torch.no_grad():
			edges = model(images)
		for edge, image_name in zip(edges, image_names):
			edge = transforms.to_pil_image(edge, mode='L')
			edge.save(os.path.join(args.save_dir, image_name))
