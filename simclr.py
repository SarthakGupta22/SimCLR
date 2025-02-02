import logging
import os

import IPython
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import accuracy, save_checkpoint, save_config_file

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.writer = SummaryWriter()
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(self.args.n_views)],
            dim=0,
        )

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        # positives represent the first column and according to this setup, target/labels will always be of class 0
        # For each image number of classes would be image - 1, so it becomes of shape (batch_size, batch_size - 1)
        logits = logits / self.args.temperature
        return logits, labels

    def save_img(self, tensor, path):
        # Convert tensor to a range [0, 1] (for float32 tensors) or [0, 255] (for int)
        tensor = (tensor - tensor.min()) / (
            tensor.max() - tensor.min()
        )  # Normalize tensor to [0, 1]
        tensor = tensor * 255  # Scale to [0, 255]

        # Convert tensor to PIL Image (from Tensor to [C, H, W] to [H, W, C] format)
        tensor = tensor.permute(
            1, 2, 0
        ).byte()  # Change shape from [C, H, W] -> [H, W, C]

        # Save the image
        image = Image.fromarray(tensor.numpy())  # Convert tensor to PIL Image
        image.save(path + ".jpg")

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):  # images: (2, batch_size, 3, H, W)

                # for i, img in enumerate(images):
                #     for j, im in enumerate(img):
                #         self.save_img(im, f"runs/image_{i}_{j}")

                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar("acc/top1", top1[0], global_step=n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate", self.scheduler.get_lr()[0], global_step=n_iter
                    )

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            )

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name),
        )
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.writer.log_dir}."
        )
