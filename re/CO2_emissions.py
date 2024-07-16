#%% md
# # CO2的排量预测，使用pytorch lightning框架示例
# ## 导入所需的库
#%%
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import yeojohnson, yeojohnson_normmax, yeojohnson_llf
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, RichProgressBar
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from sklearn.preprocessing import PowerTransformer
#%%
def create_sequencesCO2(window): 
    data = pd.read_csv('D:\Pycharm\chuan\data3.csv', encoding='gbk')
    # data = data['CO2排放总量']
    pt = PowerTransformer(method='yeo-johnson')
    data = pt.fit_transform(data['CO2排放总量'].values.reshape(-1, 1)) # 进行Yeo-Johnson变换
    # data = pt.fit_transform(data['CH4排放总量'].values.reshape(-1, 1)) # 进行Yeo-Johnson变换
    # data = pt.fit_transform(data['N2O排放总量'].values.reshape(-1, 1)) # 进行Yeo-Johnson变换
    sequences = [] 
    targets = [] 
    for i in range(len(data) - window): 
        seq = data[i:i + window] 
        target = data[i + window] 
        sequences.append(seq)
        targets.append(target)
    x=np.array(sequences)
    y=np.array(targets)
    # print(x.shape,y.shape)
    # print(x)
    return x, y, pt
#%% md
# ## 定义数据集（继承自Dataset）
#%%
class EmissionsDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.targets = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 从数据集中获取单个样本的特征和标签，并将它们转换为 PyTorch 张量（Tensor）的格式，
        # 并添加一个维度作为批处理维度。
        # 特征：前几日排放量
        # 标签：后一天的CO2排放总量
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target
#%% md
# ## pl的数据模块
#%%
class EmissionsDataModule(pl.LightningDataModule):
    def __init__(self, hparams, features, targets, pt):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.features = features
        self.targets = targets
        
        # 变换器
        self.pt = pt
        # 随机种子
        self.init_seed = hparams.init_seed

    def setup(self, stage=None): # 划分数据集
        # 调用函数，生成dataset（处理好的数据集）
        dataset = EmissionsDataset(self.features, self.targets)
        
        # 划分数据集 （划分训练、验证、测试数据集）
        # 随机种子
        train_and_val_dataset, self.test_dataset = train_test_split(dataset, test_size=0.1, random_state=self.init_seed)
        self.train_dataset, self.val_dataset = train_test_split(train_and_val_dataset, test_size=0.1, random_state=self.init_seed)
        
    # 定义数据加载器：以小批量（mini-batch）的形式提供数据，从而实现批处理训练
    # num_workers：加载数据时使用的子进程数量。这可以加速数据加载过程，特别是当数据集较大时
    # shuffle：是否在每个 epoch 开始前对数据进行随机化，通常用于训练数据。
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# 模型
from Model.model import RNNModel, GRUModel, LSTMModel, FullyConnected

#%% md
# ## pytorch lightning module
# ## 在这里，只需要将模型导入，并重写training_step, validation_step, test_step以及configure_optimizers
#%%
class EmissionsPredictor(pl.LightningModule):
    def __init__(self, hparams, pt):
        super(EmissionsPredictor, self).__init__()
        self.save_hyperparameters(hparams) # 保存超参数
        self.pt = pt
        self.model_type = hparams.model_type

        if self.model_type == 'RNN':
            self.model = RNNModel(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers)
        elif self.model_type == 'GRU':
            self.model = GRUModel(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers)
        elif self.model_type == 'LSTM':
            self.model = LSTMModel(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim,
                                   num_layers=hparams.num_layers, bidirectional=hparams.bidirectional)
        elif self.model_type == 'FC':
            self.model = FullyConnected(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim, is_linear=hparams.is_linear,
                                        window=hparams.window)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 损失函数：
        # MAE：nn.L1Loss()
        # MSE：nn.MSELoss()
        # 交叉熵：nn.CrossEntropyLoss()
        self.criterion = nn.L1Loss()

    # 前向传播
    def forward(self, x):
        # 返回选择的模型输出
        return self.model(x)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        x, y = batch
        print(x.shape)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        # 调forward（）方法得到输出，squeeze()：降维，维度为1的全部去掉；unsqueeze(0):升维
        y_hat = self(x).squeeze()
        # 计算损失：train_loss = self.nn.L1Loss(y_hat, y)
        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss, prog_bar=True, on_epoch=True)
        return {'loss' : train_loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        # print(y.shape, y_hat.shape)
        val_loss = self.criterion(y_hat, y)

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return {'val_loss' : val_loss}
        # return 1

    def test_step(self, batch, batch_idx):
        features, labels_transformed = batch
        # labels_transformed=labels_transformed.squeeze()
        outputs = self(features)
        # print(outputs.shape, labels_transformed.shape)
        # 逆变换
        original_predictions = self.pt.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
        original_labels = self.pt.inverse_transform(labels_transformed.cpu().numpy().reshape(-1, 1))
        # print(original_labels, original_predictions)
        # 计算原始尺度的MAE
        mae = mean_absolute_error(original_labels, original_predictions)
        # 计算原始尺度的R方
        r2 = r2_score(original_labels, original_predictions)
        # 计算原始尺度的Mape
        mape = mean_absolute_percentage_error(original_labels, original_predictions)

        # 记录测试损失
        self.log(self.model_type+' test_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log(self.model_type+' test_r2', r2, on_epoch=True, prog_bar=True, logger=True)
        self.log(self.model_type + ' test_mape', mape, on_epoch=True, prog_bar=True, logger=True)

        # 将mae、mape、r2记录到一个excel表中
        results = {
            'Model Type': [self.model_type],
            'Model window': [self.hparams.window],
            'test_mae': [mae],
            'test_mape': [mape],
            'test_r2': [r2]
        }
        df = pd.DataFrame(results)
        # 文件路径
        file_path = 'CO2_model_test_results.xlsx'
        # file_path = 'CH4_model_test_results.xlsx'
        # file_path = 'N2O_model_test_results.xlsx'
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            # 如果文件不存在，创建一个新的文件并写入列名和数据
            df.to_excel(file_path, index=False)
        else:
            # 如果文件存在，追加新数据
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_excel(file_path, index=False)
        return mae
    
    # # 预测
    # def predict_step(self, batch, batch_idx):
    #     features, labels_transformed = batch
    #     outputs = self(features).squeeze()
    #     return outputs
    
    # 返回Adam优化器（设置参数和学习率） 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# 画图
from Drawing.Draw import LossLogger
#%% md
# ## 定义所有超参数
#%%
def main(hparams):
    seed_everything(hparams.init_seed)  # 固定随机种子
    # df:经yeo-johnson变换、排序后的数据
    # pt：变换方法（yeo-johnson）
    # df, pt = preprocess_data(hparams.data_path) # 预处理数据并保存变换器
    # x, y, pt =create_sequences(hparams.window)
    x, y, pt = create_sequencesCO2(hparams.window)

    # 创建模型实例
    model = EmissionsPredictor(hparams=hparams, pt=pt)

    # Trainer
    earlystopping = EarlyStopping('val_loss', patience=hparams.patience, verbose=True, min_delta=0.00, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          dirpath='emissions',
                                          filename='emissions-{epoch:02d}-{val_loss:.2f}--' + hparams.model_type,
                                          save_top_k=1)

    progress_bar = TQDMProgressBar()

    # 训练过程中的损失值会记录在这个列表中
    loss_logger = LossLogger()

    callback = [earlystopping, checkpoint_callback, loss_logger, progress_bar]

    # 记录和超参数记录
    trainer = pl.Trainer(max_epochs=hparams.max_epochs, callbacks=callback, accelerator='auto', logger=True)

    # DataModule
    dm = EmissionsDataModule(hparams=hparams, features=x, targets=y, pt=pt)

    trainer.fit(model, datamodule=dm)

    best_model_path = trainer.checkpoint_callback.best_model_path
    # 测试
    trainer.test(
        model=model,
        datamodule=dm,
        # ckpt_path='D:\Pycharm\chuan\\re\emissions\emissions-epoch=99-val_loss=0.25.ckpt'
        ckpt_path=best_model_path
    )

if __name__ == '__main__':
    import argparse
    # python CO2_emissions.py --init_seed 42 --batch_size 64 --hidden_dim 128 --lr 1e-1 --num_layers 2 --num_heads 2 --max_epochs 50 --data_path "D:\Pycharm\chuan\data.csv"

    parser = argparse.ArgumentParser(description='Carbon dioxide emissions with PyTorch Lightning')
    parser.add_argument('--init_seed', type=int, default=42, help='Seed for initializing random number generators')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input features')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimensionality of hidden layers in RNN')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate for training the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads in Multi-Head Attention')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train the model')
    parser.add_argument('--data_path', type=str, default='D:\Pycharm\chuan\data3.csv', help='Path to the CSV data file')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='FC', choices=['RNN', 'GRU', 'LSTM', 'FC'], help='Type of model to use')
    parser.add_argument('--bidirectional', type=bool, default=False, help='Whether to use a bidirectional LSTM')
    parser.add_argument('--window', type=int, default=4, help='The number of features')
    parser.add_argument('--is_linear', type=bool, default=True, help='是否单线性层')
    hparams, unknown = parser.parse_known_args()

    type=['RNN', 'GRU', 'LSTM', 'FC']
    window = [4, 8, 16]
    for i in type:
        for j in window:
            hparams.model_type = i
            hparams.window = j
            main(hparams)

