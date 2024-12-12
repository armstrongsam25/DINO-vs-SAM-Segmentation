import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import jaccard_score
from segment_anything import sam_model_registry, SamPredictor
import time


# Custom Dataset for images and masks
class ImageMaskDataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.image_filenames = sorted(os.listdir(image_dir))
		self.mask_filenames = sorted(os.listdir(mask_dir))
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, idx):
		# Load image and mask
		image_path = os.path.join(self.image_dir, self.image_filenames[idx])
		mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

		# Open .tif image and convert to RGB
		image = Image.open(image_path).convert("RGB")

		# Open .png mask and convert to grayscale
		mask = Image.open(mask_path).convert("L")

		# Apply transforms if defined
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			mask = self.target_transform(mask)

		return image, mask


class SAMFeatureExtractor(torch.nn.Module):
	def __init__(self, sam_model):
		super(SAMFeatureExtractor, self).__init__()
		self.sam_model = sam_model

	def forward(self, x):
		# Extract image embeddings
		with torch.no_grad():
			image_embeddings = self.sam_model.image_encoder(x)
		return image_embeddings


class LinearClassifierToken(torch.nn.Module):
	def __init__(self, embedding_dim, nc=1, tokenW=32, tokenH=32):
		super(LinearClassifierToken, self).__init__()
		self.W = tokenW
		self.H = tokenH
		self.nc = nc
		self.conv = torch.nn.Conv2d(embedding_dim, nc, (1, 1))

	def forward(self, x):
		# Reshape embeddings to match expected input
		return self.conv(x)


# Function to calculate IoU for a single batch
def calculate_iou(preds, target, threshold=0.5):
	"""
	Calculate the Intersection over Union (IoU) for a batch of predictions.
	Args:
		preds (tensor): The predicted logits (batch_size, 1, H, W)
		target (tensor): The ground truth (batch_size, 1, H, W)
		threshold (float): Threshold to classify pixel as 1 (object) or 0 (background)
	"""
	# Apply sigmoid to get probabilities, then threshold to binary (0 or 1)
	preds = (torch.sigmoid(preds) > threshold).float()  # Convert to binary (0 or 1)

	# Flatten the predictions and target for IoU calculation
	preds = preds.view(-1).cpu().numpy()  # Flatten and move to CPU
	target = target.view(-1).cpu().numpy()  # Flatten and move to CPU

	# Ensure both preds and target are of integer type (0 or 1)
	preds = preds.astype(np.int32)
	target = target.astype(np.int32)

	# Compute IoU using Jaccard index (sklearn)
	return jaccard_score(target, preds)


if __name__ == '__main__':
	# Initialize SAM model
	sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
	model_type = "vit_b"
	device = "cuda"

	sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

	# Feature extractor and classifier
	feature_extractor = SAMFeatureExtractor(sam).cuda()
	classlayer = LinearClassifierToken(embedding_dim=256, nc=1, tokenW=32, tokenH=32).cuda()

	optimizer = torch.optim.Adam(classlayer.parameters())
	lossfn = torch.nn.BCEWithLogitsLoss()

	# Directories for images and masks
	train_image_dir = "./data/images/train"
	train_mask_dir = "./data/masks/train"
	val_image_dir = "./data/images/val"
	val_mask_dir = "./data/masks/val"

	# Define transformations for images and masks
	image_transform = transforms.Compose([
		transforms.Resize((1024, 1024)),  # SAM typically works with 1024x1024
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	mask_transform = transforms.Compose([
		transforms.Resize((1024, 1024)),  # Match image size
		transforms.ToTensor(),  # Convert to tensor
	])

	# Create dataset and dataloader
	train_dataset = ImageMaskDataset(train_image_dir, train_mask_dir,
									 transform=image_transform,
									 target_transform=mask_transform)
	val_dataset = ImageMaskDataset(val_image_dir, val_mask_dir,
								   transform=image_transform,
								   target_transform=mask_transform)

	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

	# Define the learning rate scheduler
	scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

	# Set number of epochs
	num_epochs = 30
	patience = 5  # Early stopping patience
	best_val_loss = float('inf')
	epochs_without_improvement = 0

	start_time = time.time()

	for epoch in range(num_epochs):
		# Training loop
		classlayer.train()
		for batch_idx, data in enumerate(train_dataloader):
			images, masks = data
			images, masks = images.cuda(), masks.cuda()

			# Downsample masks to match model output size (32x32)
			target = torch.nn.functional.interpolate(masks, size=(64, 64), mode='bilinear', align_corners=False)

			# Extract features using SAM's image encoder
			features = feature_extractor(images)

			# Make predictions
			preds = classlayer(features)

			# Compute loss
			loss = lossfn(preds, target)

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Train Loss: {loss.item():.4f}")

		# Validation loop
		classlayer.eval()
		val_loss = 0.0
		iou_list = []
		with torch.no_grad():
			for batch_idx, data in enumerate(val_dataloader):
				images, masks = data
				images, masks = images.cuda(), masks.cuda()

				# Downsample masks to 32x32
				target = torch.nn.functional.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)

				# Extract features and compute predictions
				features = feature_extractor(images)
				preds = classlayer(features)

				# Compute loss
				loss = lossfn(preds, target)
				val_loss += loss.item()

				# Calculate IoU for the batch and add to the list
				iou_batch = calculate_iou(preds, target)
				iou_list.append(iou_batch)

		mean_iou = np.mean(iou_list)
		val_loss /= len(val_dataloader)

		print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, mIoU: {mean_iou:.4f}")

		# Step the scheduler
		scheduler.step(val_loss)

		# Early stopping check
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= patience:
			print(f"Early stopping triggered after {epoch + 1} epochs")
			break

	print(f'Time taken: {time.time() - start_time}')
