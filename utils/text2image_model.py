import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


# class Generator(nn.Module):
#     def __init__(self,
#             batch_size=64,
#             image_size=80,
#             z_dim,
#             text_embedding_dim=1000,
#             text_reduced_dim):
#         super(Generator, self).__init__()

#         self.image_size = image_size
# 		self.z_dim = z_dim
# 		self.text_embedding_dim = text_embedging_dim

# 		self.concat = nn.Linear(z_dim + text_reduced_dim, 64 * 8 * 4 * 4).cuda()
# 		self.text_reduced_dim = nn.Linear(text_embed_dim, text_reduced_dim).cuda()
		
# 		self.network = nn.Sequential(
# 			nn.ReLU(),
# 			nn.ConvTranspose2d(512, 256, 4, 2, 1),
# 			nn.BatchNorm2d(256),
# 			nn.ReLU(),
# 			nn.ConvTranspose2d(256, 128, 4, 2, 1),
# 			nn.BatchNorm2d(128),
# 			nn.ReLU(),
# 			nn.ConvTranspose2d(128, 64, 4, 2, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),
# 			nn.ConvTranspose2d(64, 3, 4, 2, 1),
# 			nn.Tanh()
# 		).cuda()

#         def forward(self, text_embedding, z):
#             reduced_text = self.text_reduced_dim(text_embedding.cuda())  # (batch_size, text_reduced_dim)
#             concat = torch.cat((reduced_text, z.cuda()), 1)              # (batch_size, text_reduced_dim + z_dim)
#             concat = self.concat(concat)                                 # (batch_size, 64*8*4*4)
#             concat = concat.view(-1, 4, 4, 64 * 8)                       # (batch_size, 4, 4, 64*8)
        
#             concat = concat.permute(0, 3, 1, 2)                          # (batch_size, 512, 4, 4)
#             output = self.network(concat)                                # (batch_size, 3, 64, 64)
#             output = output.permute(0, 2, 3, 1)                          # (batch_size, 64, 64, 3)
            
#             output = output / 2. + 0.5                                   # (batch_size, 64, 64, 3)

#             return output


# class Discriminator(nn.Module):
# 	def __init__(self, batch_size=64,
#             image_size=80,
#             text_embedding_dim,
#             text_reduced_dim):
# 		super(Discriminator, self).__init__()

# 		self.batch_size = batch_size
# 		self.image_size = image_size
# 		self.in_channels = image_size[2]
# 		self.text_embedding_dim = text_embedding_dim
# 		self.text_reduced_dim = text_reduced_dim

# 		self.network = nn.Sequential(
# 			nn.Conv2d(self.in_channels, 64, 4, 2, 1, bias=False),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
# 			nn.BatchNorm2d(128),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
# 			nn.BatchNorm2d(256),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(0.2, inplace=True)).cuda()

# 		# output_dim = (batch_size, 4, 4, 512)
# 		# text.size() = (batch_size, text_embed_dim)

# 		self.cat_network = nn.Sequential(
# 			nn.Conv2d(512 + self.text_reduced_dim, 512, 4, 2, 1, bias=False),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(0.2, inplace=True)).cuda()

# 		self.text_reduced_dim = nn.Linear(self.text_embedding_dim, 
#                                           self.text_reduced_dim).cuda()
		
# 		self.linear = nn.Linear(2 * 2 * 512, 1).cuda()

# 	def forward(self, image, text):
# 		image = image.permute(0, 3, 1, 2)                     # (batch_size, 3, 64, 64)
# 		output = self.network(image)                          # (batch_size, 512, 4, 4)
# 		output = output.permute(0, 2, 3, 1)                   # (batch_size, 4, 4, 512)
		
# 		text_reduced = self.text_reduced_dim(text)            # (batch_size, text_reduced_dim)
# 		text_reduced = text_reduced.unsqueeze(1)              # (batch_size, 1, text_reduced_dim)
# 		text_reduced = text_reduced.unsqueeze(2)              # (batch_size, 1, 1, text_reduced_dim)
# 		text_reduced = text_reduced.expand(-1, 4, 4, -1)

# 		concat_out = torch.cat((d_net_out, text_reduced), 3)  # (1, 4, 4, 512+text_reduced_dim)
		
# 		logit = self.cat_network(concat_out.permute(0, 3, 1, 2))
# 		logit = logit.reshape(-1, logit.size()[1] * logit.size()[2] * logit.size()[3])
# 		logit = self.linear(logit)

# 		output = torch.sigmoid(logit)

# 		return output, logit


class VAE(nn.Module):
    def __init__(self, 
            text_size,
            image_size,
            hidden_size,
            latent_size,
            learning_rate):
        super(VAE, self).__init__()

        self.hidden_size = hidden_size

        self.encode_1 = nn.Linear(text_size, hidden_size)
        self.encode_2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_size) # mean
        self.log_var = nn.Linear(hidden_size, latent_size) # std

        self.decode_1 = nn.Linear(latent_size, hidden_size)
        self.decode_2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 128, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss(reduction='sum')
        

    def encode(self, inputs):
        outputs = self.dropout(self.encode_1(inputs))
        outputs = self.relu(outputs)
        outputs = self.dropout(self.encode_2(outputs))
        outputs = self.relu(outputs)

        return outputs


    def gaussian_param_projection(self, inputs):
        mean = self.mean(inputs)
        log_var = self.log_var(inputs)

        return mean, log_var
    

    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(log_var / 2)
            eps = torch.randn_like(std)

            return mean + eps * std
        else: 
            return mean


    def decode(self, z):
        outputs = self.dropout(self.decode_1(z))
        outputs = self.relu(outputs)

        outputs = outputs.view(outputs.size(0), self.hidden_size, 1, 1)
        outputs = self.decode_2(outputs)
        # outputs = self.relu(outputs)
        
        # outputs = self.sigmoid(outputs)

        return outputs
    

    def forward(self, inputs):
        outputs = self.encode(inputs)
        mean, log_var = self.gaussian_param_projection(outputs)
        z = self.reparameterize(mean, log_var)
        inputs_reconstruction = self.decode(z)

        return inputs_reconstruction, mean, log_var


    def KL_div(self, mean, log_var):
        return - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())


    def loss_function(self, reconstruction, target, mean, log_var):
        target = target.reshape((target.size(0), 3, 80, 80))

        batch_BCE = self.criterion(reconstruction, target)
        batch_KL = self.KL_div(mean=mean, log_var=log_var)

        return batch_BCE + batch_KL


    def compute_loss(self, x_data, image_target):
        with torch.no_grad():
            image_target = image_target.view((image_target.size(0), 3, 80, 80))

            reconstruction, mean, log_var = self.forward(x_data)

            return self.loss_function(reconstruction, image_target, mean, log_var).item()


    def train_model(self, batch_x, batch_image):
        reconstruction, mean, log_var = self.forward(batch_x)
        loss = self.loss_function(reconstruction=reconstruction,
                                  target=batch_image,
                                  mean=mean,
                                  log_var=log_var)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, x_data):
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x_data)

        return reconstruction


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()


class MultilabelClassification(nn.Module):
    def __init__(self,
            n_components,
            hidden_size=4096,
            num_classes=43,
            learning_rate=0.0001):
        super(MultilabelClassification, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2))

        self.linear_1 = nn.Linear(128*5*5+n_components, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input_data, input_image):
        # input_image = input_image.view((input_image.size(0), 3, 80, 80))
        output = self.cnn(input_image)

        # print(output.size())
        output = output.view((output.size(0), -1))
        # print(output.size(), input_data.size())
        output = torch.cat((output, input_data), 1)
        output = self.relu(self.dropout(self.linear_1(output)))
        output = self.sigmoid(self.linear_2(output))

        return output


    def train_model(self, batch_x, batch_image, batch_y):
        output = self.forward(input_data=batch_x, input_image=batch_image)
        loss = self.criterion(output, batch_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def compute_loss(self, x_data, image, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data, input_image=image)

            return F.binary_cross_entropy(output, y_target).item()


    def evaluate(self, x_data, image, y_target):
        with torch.no_grad():
            output = self.forward(input_data=x_data, input_image=image)
            output = torch.argsort(output, dim=1, descending=True)

            predictions = [np.where(out < 6)[0] for out in output]
            target = [np.where(y == 1)[0] for y in y_target]

            intersections = [set(target[i]).intersection(set(predictions[i])) \
                                for i in range(len(target))]

            recall = [len(intersections[i]) / len(target[i]) \
                        for i in range(len(target))]

            return np.mean(recall)


    def predict(self, x_data, image, categories):
        with torch.no_grad():
            output = self.forward(intput_data=x_data, input_image=image)
            output = torch.argsort(output, dim=1, descending=True)
            predictions = [categories[np.where(out < 6)[0]] for out in output]

        return predictions


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.eval()