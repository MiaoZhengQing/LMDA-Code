from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import logging
import torch.nn as nn
import matplotlib.pyplot as plt


def setup_seed(seed=521):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EEGDataLoader(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)  # label without one-hot coding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        label_tensor = self.labels[idx]
        return data_tensor, label_tensor


class Measurement(object):

    def __init__(self, test_df, classes):
        '''
        :param test_df: DataFrame of test, 将其转化为ndarray似乎更方便;
        :return: None
        '''
        self.test_df = test_df.values
        self.classes = classes

    def max_acc(self):  # using kappa for MI, AUC for P300
        return self.test_df.max()

    def last10_acc(self):
        return self.test_df[-10:].mean()

    def last10_std(self):
        return self.test_df[-10:].std()



class Experiment(object):
    def __init__(self, model, optimizer, train_dl, test_dl, val_dl, fig_path, device='cuda:0',
                 step_one=300, model_constraint=None,
                 p300_flag=False, imbalanced_class=1, classes=2,
                 ):
        self.model = model
        self.optim4model = optimizer
        self.model_constraint = model_constraint
        self.best_test = float("-inf")
        self.step_one_epoch = step_one
        self.device = device
        # for imbalanced classed
        self.classes = classes
        class_weight = [1.0] * classes
        class_weight[0] *= imbalanced_class  # kaggle ERN, 0:1=3:7, 因此imbalanced_weight=3/7
        class_weight = torch.FloatTensor(class_weight).to(device)
        self.nllloss = nn.CrossEntropyLoss(weight=class_weight)
        self.datasets = OrderedDict((("train", train_dl), ("valid", val_dl), ("test", test_dl)))
        if val_dl is None:
            self.datasets.pop("valid")
        if test_dl is None:
            self.datasets.pop("test")
        self.fig_path = fig_path
        # initialize epoch dataFrame instead of loss and acc for train and test
        self.val_df = pd.DataFrame()  
        self.train_df = pd.DataFrame()  
        self.epoch_df = pd.DataFrame()  
        # p300
        self.p300_flag = p300_flag

    def train_epoch(self):
        self.model.train()
        batch_size, cls_loss, train_acc, all_labels = [], [], [], []
        for i, (train_input, train_target) in enumerate(self.datasets['train']):
            train_input = train_input.to(self.device).float()
            train_label = train_target.to(self.device).long()
            source_softmax = self.model(train_input)
            nll_loss = self.nllloss(source_softmax, train_label)  # with margin

            batch_size.append(len(train_input))
            if self.p300_flag:  # AUC instead of acc
                # get AUC
                source_softmax = nn.Softmax(dim=0)(source_softmax)  
                batch_acc = source_softmax.cpu().detach().numpy()[:, 1]  
                all_labels.append(train_label.cpu().detach().numpy())
            else:
                _, predicted = torch.max(source_softmax.data, 1)
                batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(
                    train_label)
            train_acc.append(batch_acc)
            cls_loss_np = nll_loss.cpu().detach().numpy()
            cls_loss.append(cls_loss_np)

            self.optim4model.zero_grad()
            nll_loss.backward()
            self.optim4model.step()

            if self.model_constraint is not None:
                self.model_constraint.apply(self.model)

        if self.p300_flag:
            train_acc = np.concatenate(train_acc, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            epoch_acc = roc_auc_score(all_labels, train_acc)
        else:
            epoch_acc = sum(train_acc) / len(train_acc) * 100
        epoch_loss = sum(cls_loss) / len(cls_loss)

        train_dicts_per_epoch = OrderedDict()

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)

        self.train_df = self.train_df.append(train_dicts_per_epoch, ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  # 让epoch_df中的顺序和row_dict中的一致

    def test_batch(self, input, target):
        self.model.eval()
        with torch.no_grad():
            val_input = input.to(self.device).float()
            val_target = target.to(self.device).long()
            val_fc1 = self.model(val_input)
            loss = self.nllloss(val_fc1, val_target)
            if self.p300_flag:
                source_softmax = nn.Softmax(dim=0)(val_fc1)  
                preds = source_softmax.cpu().detach().numpy()[:, 1]  
            else:
                _, preds = torch.max(val_fc1.data, 1)  
                preds = preds.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
        return preds, loss

    def monitor_epoch(self, datasets):
        datasets['test'] = self.datasets['test']
        result_dicts_per_monitor = OrderedDict()  
        for setname in datasets:  
            assert setname != 'train', 'dataset without train set'
            assert setname in ["test"]  
            dataset = datasets[setname]
            batch_size, epoch_loss, epoch_acc, test_labels = [], [], [], []
            for i, (input, target) in enumerate(dataset):  
                pred, loss = self.test_batch(input, target)  
                epoch_loss.append(loss)
                batch_size.append(len(target))
                if self.p300_flag:  
                    epoch_acc.append(pred)
                    test_labels.append(target.cpu().detach().numpy())
                else:
                    epoch_acc.append(np.equal(pred, target.numpy()).sum())  
            if self.p300_flag:
                epoch_acc = np.concatenate(epoch_acc, axis=0)
                test_labels = np.concatenate(test_labels, axis=0)
                epoch_acc = roc_auc_score(test_labels, epoch_acc)
            else:
                epoch_acc = sum(epoch_acc) / sum(batch_size) * 100  
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            key_loss = setname + '_loss'
            key_acc = setname + '_acc'
            loss = {key_loss: epoch_loss}
            acc = {key_acc: epoch_acc}
            result_dicts_per_monitor.update(loss)
            result_dicts_per_monitor.update(acc)
            if epoch_acc > self.best_test:
                self.best_test = epoch_acc
                logging.info("New best test acc %5f", epoch_acc)
        self.val_df = self.val_df.append(result_dicts_per_monitor, ignore_index=True)
        self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]  # 让epoch_df中的顺序和row_dict中的一致

    def train_step_one(self):

        logging.info("****** Run until first stop... ********")
        logging.info("Train size: %d", len(self.datasets['train']))
        logging.info("Test size: %d", len(self.datasets['test']))

        epoch = 0
        while epoch < self.step_one_epoch:  # 设置训练终止条件
            self.train_epoch()
            self.monitor_epoch({})
            self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)
   
            if epoch % 20 == 0 or epoch > self.step_one_epoch - 50:
                self.log_epoch()
            epoch += 1

    def log_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.epoch_df) - 1
        logging.info("Epoch {:d}".format(i_epoch))
        last_row = self.epoch_df.iloc[-1]
        for key, val in last_row.iteritems():
             logging.info("%s       %.5f", key, val)
        logging.info("")

    def run(self):
        self.train_step_one()
        # logging.info('Best test acc %.5f ', self.epoch_df['test_acc'].max())
        self.measure = Measurement(self.epoch_df['test_acc'], classes=self.classes)
        if self.epoch_df['test_acc'].max() < 1:  # in p300, using AUC instead of acc
            logging.info('Best test AUC %.5f ', self.measure.max_acc())
            logging.info('mean AUC of last 10 epochs %.5f ', self.measure.last10_acc())
            logging.info('std of last 10 epochs %.5f ', self.measure.last10_std())
            logging.info('offset of max and mean AUC (percentage) %.5f ', self.measure.max_mean_offset())
        else:
            logging.info('Best test acc %.5f ', self.measure.max_acc())
            logging.info('mean acc of last 10 epochs %.5f ', self.measure.last10_acc())
            logging.info('std of last 10 epochs %.5f ', self.measure.last10_std())
            logging.info('offset: max SUB mean acc %.5f ', self.measure.max_mean_offset())
        logging.info('* '*20)
        logging.info('Index score(0.4*max+0.4*mean-0.2*offset): %.5f', self.measure.index_equation())
        self.save_acc_loss_fig()

    def save_acc_loss_fig(self):

        test_acc = self.epoch_df['test_acc'].values.tolist()
        plt.figure()
        plt.plot(range(len(test_acc)), test_acc, label='test acc', linewidth=0.7)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')
        plt.savefig(self.fig_path + 'mean{:.3f}max{:.3f}.png'.format(self.measure.last10_acc(), self.measure.max_acc()))


