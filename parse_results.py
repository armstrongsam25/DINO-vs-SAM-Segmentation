import re
import matplotlib.pyplot as plt

# Read the text file
file_paths = ['results/raw_results_dino.txt', 'results/raw_results_sam.txt']

for file_path in file_paths:
	with open(file_path, 'r') as file:
		lines = file.readlines()

	# Initialize lists to store epoch numbers, validation loss, and mIoU
	epochs = []
	validation_loss = []
	miou = []

	# Parse the lines to extract data
	for line in lines:
		# Match validation loss and mIoU
		match = re.search(r'Epoch (\d+), Validation Loss: ([\d.]+), mIoU: ([\d.]+)', line)
		if match:
			epoch = int(match.group(1))
			val_loss = float(match.group(2))
			miou_value = float(match.group(3))

			epochs.append(epoch)
			validation_loss.append(val_loss)
			miou.append(miou_value)

	# Plot the results
	plt.figure(figsize=(10, 5))

	# Plot Validation Loss
	plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
	# Plot mIoU
	plt.plot(epochs, miou, label='mIoU', marker='s')

	# Customize the plot
	if file_path == 'results/raw_results_dino.txt':
		plt.title('Validation Loss and mIoU (DINOv2)')
	else:
		plt.title('Validation Loss and mIoU (SAM)')
	plt.xlabel('Epoch')
	plt.ylabel('Metric Value')
	plt.legend()
	plt.grid(True)

	# Show the plot
	plt.show()


# Function to parse a text file and extract epochs and mIoU
def parse_miou(file_path):
	epochs = []
	miou = []

	with open(file_path, 'r') as file:
		lines = file.readlines()

		for line in lines:
			# Match validation loss and mIoU
			match = re.search(r'Epoch (\d+), Validation Loss: [\d.]+, mIoU: ([\d.]+)', line)
			if match:
				epoch = int(match.group(1))
				miou_value = float(match.group(2))

				epochs.append(epoch)
				miou.append(miou_value)

	return epochs, miou


def parse_loss(file_path):
	epochs = []
	val_loss = []

	with open(file_path, 'r') as file:
		lines = file.readlines()

		for line in lines:
			# Match validation loss and mIoU
			match = re.search(r'Epoch (\d+), Validation Loss: ([\d.]+), mIoU: [\d.]+', line)
			if match:
				epoch = int(match.group(1))
				loss_value = float(match.group(2))

				epochs.append(epoch)
				val_loss.append(loss_value)

	return epochs, val_loss


# Parse the two files
file1 = 'results/raw_results_dino.txt' # Replace with the path to your first file
file2 = 'results/raw_results_sam.txt'  # Replace with the path to your second file

epochs1, miou1 = parse_miou(file1)
epochs2, miou2 = parse_miou(file2)

# Plot mIoU for both models
plt.figure(figsize=(10, 5))

# Model 1
plt.plot(epochs1, miou1, label='DINOv2 mIoU', marker='o', linestyle='-')
# Model 2
plt.plot(epochs2, miou2, label='SAM mIoU', marker='s', linestyle='--')

# Customize the plot
plt.title('mIoU Comparison Between Models')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

epochs1, val_loss1 = parse_loss(file1)
epochs2, val_loss2 = parse_loss(file2)

# Plot validation loss for both models
plt.figure(figsize=(10, 5))

# Model 1
plt.plot(epochs1, val_loss1, label='DINOv2 Validation Loss', marker='o', linestyle='-')
# Model 2
plt.plot(epochs2, val_loss2, label='SAM Validation Loss', marker='s', linestyle='--')

# Customize the plot
plt.title('Validation Loss Comparison Between Models')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
