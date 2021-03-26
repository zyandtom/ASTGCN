# -*- coding:utf-8 -*-

import os
import shutil
import time
from datetime import datetime
import configparser
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from lib.data_preparation import read_and_generate_dataset
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from torch.optim import lr_scheduler
from sklearn import metrics
from lib.utils import *

graph_signal_matrix_filename = 'data/PEMS08/pems08.npz'
num_of_weeks = 1
num_of_days = 1
num_of_hours = 3
num_for_predict = 12
points_per_hour = 12
merge = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # train_week = torch.from_numpy(all_data['train']['week'].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    # train_day = torch.from_numpy(all_data['train']['day'].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    train_recent = torch.from_numpy(all_data['train']['recent'].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    train_target = torch.from_numpy(all_data['train']['target']).type(torch.FloatTensor)
    val_recent = torch.from_numpy(all_data['val']['recent'].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    val_target = torch.from_numpy(all_data['val']['target']).type(torch.FloatTensor)
    test_recent = torch.from_numpy(all_data['test']['recent'].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    test_target = torch.from_numpy(all_data['test']['target']).type(torch.FloatTensor)
    print(train_recent.shape)
    print(train_target.shape)
    print(val_recent.shape)
    print(val_target.shape)
    print(test_recent.shape)
    print(test_target.shape)

    # definition of dataset
    # train_dataset_week = TensorDataset(train_week, train_target)
    # train_dataset_day = TensorDataset(train_day, train_target)
    train_dataset_recent = TensorDataset(train_recent, train_target)
    val_dataset_recent = TensorDataset(val_recent, val_target)
    test_dataset_recent = TensorDataset(test_recent, test_target)

    # define the dataloader
    train_loader_recent = DataLoader(dataset=train_dataset_recent, shuffle=True, batch_size=128, num_workers=4)
    val_loader_recent = DataLoader(dataset=val_dataset_recent, shuffle=False, batch_size=128, num_workers=4)
    test_loader_recent = DataLoader(dataset=test_dataset_recent, shuffle=False, batch_size=128, num_workers=4)

    class lstm(nn.Module):
        def __init__(self):
            super(lstm, self).__init__()
            self.conv = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(1, 3), stride=1)
            self.rnn = torch.nn.LSTM(
                input_size=170,
                hidden_size=170,
                num_layers=1,
                batch_first=True,
                dropout=0.5
            )
            self.out=torch.nn.Linear(in_features=170,out_features=170*12)

        def forward(self, x):
            x = self.conv(x)
            # print(x.shape)
            x = x.reshape(-1, 36, 170)
            # print(x.shape)
            output, (h_n, c_n) = self.rnn(x)
            # print(output.shape)
            output_in_last_timestep = output[:, -1, :]
            # output_in_last_timestep=h_n[-1,:,:]
            # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
            x = self.out(output_in_last_timestep)
            x = x.reshape(-1, 170, 12)
            return x

    def get_time_dif(start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def evaluate(model, data_loader):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, outputs.data.cpu().numpy())
        mse = mean_squared_error(labels_all, predict_all)**0.5
        return mse, loss_total / (len(data_loader)*128)

    def train(model, train_loader, val_loader, test_loader, learning_rate=0.1,
              num_epochs=50):
        start_time = time.time()
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        total_batch = 0
        val_best_loss = float('inf')
        last_improve = 0
        flag = False

        model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            print('Epoch[{}/{}]'.format(epoch + 1, num_epochs))
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                # print("output shape: ", outputs.shape)
                # print("label shape: ", labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    true = labels.data.cpu().numpy()
                    train_mse = mean_squared_error(true, outputs.data.cpu().numpy())**0.5
                    val_mse, val_loss = evaluate(model, val_loader)
                    if val_loss < val_best_loss:
                        val_best_loss = val_loss
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss:{1:.2f}, Train mse:{2:.2f}, Val Loss:{3:.2f}, Val mse:{4:.2f}, Time:{5} {6}'
                    print(msg.format(total_batch + 1, running_loss / (10*128), train_mse, val_loss, val_mse, time_dif, improve))
                    model.train()
                    running_loss = 0.0
                total_batch += 1

            #test data
            model.eval()
            predict_all = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    predict_all.append(outputs.data.cpu().numpy())
            predict_all = np.concatenate(predict_all, 0)
            predict_all = (predict_all.transpose((0, 2, 1))
                          .reshape(predict_all.shape[0], -1))

            prediction_path = os.path.join('result_lstm', 'ASTGCN_prediction_08' + '_epoch%s' % (epoch))

            np.savez_compressed(
                os.path.normpath(prediction_path),
                prediction=predict_all
            )
            model.train()

    model = lstm().to(device)
    train(model, train_loader_recent, val_loader_recent, test_loader_recent)


