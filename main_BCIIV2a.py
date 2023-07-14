from compared_models import weights_init, MaxNormDefaultConstraint, EEGNet, ShallowConvNet
from lmda_model import LMDA
from experiment import EEGDataLoader, Experiment, setup_seed
# dataloader and preprocess
from data_loader import BCICompetition4Set2A, extract_segment_trial
from data_preprocess import preprocess4mi, mne_apply, bandpass_cnt
# tools for pytorch
from torch.utils.data import DataLoader
import torch
# tools for numpy as scipy and sys
import logging
import os
import time
import datetime
# tools for plotting confusion matrices and t-SNE
from torchsummary import summary


# ========================= BCIIV2A data =====================================

def bci4_2a():  # 通用模型模块, 不需要更改, 放在main中方便调试;
    dataset = 'BCI4_2A'
    data_path = "/home/dog/Documents/EEGDataSet/BCICIV_2a_gdf/"

    train_filename = "{}T.gdf".format(subject_id)
    test_filename = "{}E.gdf".format(subject_id)
    train_filepath = os.path.join(data_path, train_filename)
    test_filepath = os.path.join(data_path, test_filename)
    train_label_filepath = train_filepath.replace(".gdf", ".mat")
    test_label_filepath = test_filepath.replace(".gdf", ".mat")

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath
    )
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
    )
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # band-pass before segment trials
    # train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    # test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)

    train_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                 filt_order=200, fs=250, zero_phase=False),
                          train_cnt)

    test_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                filt_order=200, fs=250, zero_phase=False),
                         test_cnt)

    train_data, train_label = extract_segment_trial(train_cnt)
    test_data, test_label = extract_segment_trial(test_cnt)

    train_label = train_label - 1
    test_label = test_label - 1

    preprocessed_train = preprocess4mi(train_data)
    preprocessed_test = preprocess4mi(test_data)

    train_loader = EEGDataLoader(preprocessed_train, train_label)
    test_loader = EEGDataLoader(preprocessed_test, test_label)

    train_dl = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    test_dl = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    valid_dl = None

    model_id = '%s' % share_model_name
    folder_path = './%s/%s/' % (dataset, subject_id)  # mkdir in current folder, and name it by target's num
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
    output_file = os.path.join(folder_path, '%s%s.log' % (model_id, code_num))
    fig_path = folder_path + str(model_id) + code_num  # 用来代码命名
    logging.basicConfig(
        datefmt='%Y/%m/%d %H:%M:%S',
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=output_file,
    )
    # log.info自带format, 因此只需要使用响应的占位符即可.
    logging.info("****************  %s for %s! ***************************", model_id, subject_id)

    if share_model_name == 'LMDA':
        Net = LMDA(num_classes=mi_class, chans=channels, samples=samples,
                   channel_depth1=model_para['channel_depth1'],
                   channel_depth2=model_para['channel_depth2'],
                   kernel=model_para['kernel'], depth=model_para['depth'],
                   ave_depth=model_para['pool_depth'], avepool=model_para['avepool'],
                   ).to(device)
        logging.info(model_para)

    elif share_model_name == 'EEGNet':
        Net = EEGNet(num_classes=mi_class, chans=channels, samples=samples).to(device)

    else:  # ConvNet
        Net = ShallowConvNet(num_classes=mi_class, chans=channels, samples=samples).to(device)

    Net.apply(weights_init)
    Net.apply(weights_init)

    logging.info(summary(Net, show_input=False))

    model_optimizer = torch.optim.AdamW(Net.parameters(), lr=lr_model)
    model_constraint = MaxNormDefaultConstraint()
    return train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path


if __name__ == "__main__":
    start_time = time.time()
    setup_seed(521)  # 521, 322
    print('* * ' * 20)

    # basic info of the dataset
    mi_class = 4
    channels = 22
    samples = 1125
    sample_rate = 250

    # subject of the datase
    subject_id = 'A06'
    device = torch.device('cuda:2')

    print('subject_id: ', subject_id)

    model_para = {
        'channel_depth1': 24,  # 推荐时间域的卷积层数比空间域的卷积层数更多
        'channel_depth2': 9,
        'kernel': 75,
        'depth': 9,
        'pool_depth': 1,
        'avepool': sample_rate // 10,  # 还是推荐两步pooling的
        'avgpool_step1': 1,
    }

    share_model_name = 'LMDA'
    assert share_model_name in ['LMDA', 'EEGNet', 'ConvNet']

    today = datetime.date.today().strftime('%m%d')
    if share_model_name == 'LMDA':

        code_num = 'D{depth}_D{depth1}_D{depth2}_pool{pldp}'.format(depth=model_para['depth'],
                                                                    depth1=model_para['channel_depth1'],
                                                                    depth2=model_para['channel_depth2'],
                                                                    pldp=model_para['avepool'] * model_para[
                                                                        'avgpool_step1'])
    else:
        code_num = ''
    print(share_model_name + code_num)
    print(device)
    print(model_para)
    print('* * ' * 20)

    # ===============================  超 参 数 设 置 ================================
    lr_model = 1e-3
    step_one_epochs = 300
    batch_size = 32
    kwargs = {'num_workers': 1, 'pin_memory': True}  # 因为pycharm开启了多进行运行main, num_works设置多个会报错
    # ================================================================================

    train_dl, valid_dl, test_dl, ShallowNet, model_optimizer, model_constraint, fig_path = bci4_2a()

    exp = Experiment(model=ShallowNet,
                     device=device,
                     optimizer=model_optimizer,
                     train_dl=train_dl,
                     test_dl=test_dl,
                     val_dl=valid_dl,
                     fig_path=fig_path,
                     model_constraint=model_constraint,
                     step_one=step_one_epochs,
                     classes=mi_class,
                     )
    exp.run()

    end_time = time.time()
    # print('Net channel weight:', ShallowNet.channel_weight)
    logging.info('Param again {}'.format(model_para))
    logging.info('Done! Running time %.5f', end_time - start_time)
