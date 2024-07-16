import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

class LossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # 记录训练损失
        self.train_losses.append(trainer.callback_metrics['train_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # 记录验证损失
        self.val_losses.append(trainer.callback_metrics['val_loss'].item())

    def on_train_end(self, trainer, pl_module):
        # 调整列表长度以确保一致性
        min_length = min(len(self.train_losses), len(self.val_losses))
        self.train_losses = self.train_losses[:min_length]
        self.val_losses = self.val_losses[:min_length]

        # 绘制训练和验证损失
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
