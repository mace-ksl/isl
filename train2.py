import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
import timm
import data_set
import os
import pytorch_lightning as pl
import model
path = os.path.join(os.getcwd(), "data_set")
data = data_set.DataSet(path)

data_train_in = data.get_input_images_as_array("split_cy5", "train")
data_train_out = data.get_output_images_as_array("split_cy5", "train")

data_val_in = data.get_input_images_as_array("split_cy5", "val")
data_val_out = data.get_output_images_as_array("split_cy5", "val")

data_test_in = data.get_input_images_as_array("split_cy5", "test")
data_test_out = data.get_output_images_as_array("split_cy5", "test")

train_loader, val_loader, test_loader = data.create_torch_data_loader(
    data_train_in, data_train_out, data_val_in, data_val_out, data_test_in, data_test_out
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your model and optimizer
#model = timm.create_model('resnet50', pretrained=True)
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 1)  # Regression task has a single output
model = model.Model("Unet", "mit_b2", in_channels=3, out_classes=1).to(device)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
log_interval = 50
# Define loss function for regression
criterion = nn.MSELoss()
#criterion = torch.nn.CrossEntropyLoss()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Define scaler for automatic scaling of gradients
scaler = GradScaler()
num_epochs = 10

loss_values = []
# Training loop
for epoch in range(num_epochs):
    for input_image, target_mask in train_loader:
        optimizer.zero_grad()
        

        # Forward pass
        predictions = model(input_image)
        predictions = predictions.to(device)
        target_mask = target_mask.to(device)

        # Calculate regression loss
        loss = criterion(predictions, target_mask)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()
        #optimizer.zero_grad()
        #print(f"Epoch {epoch} ------- Loss {loss} ")
    # Optionally, print or log the loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    loss_values.append(loss.item())

torch.save(model.state_dict(), 'E:/Data_sets/Github/timm/isl/test_model.pth')

import matplotlib.pyplot as plt
# Plot the loss curve
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
