import warnings
import torch


class Default(object):
    load_model_path = None  # load model path

    # visualization
    vis_env = 'main'  # visdom env
    vis_port = 8097  # visdom port
    flag = 'mir'
    
    batch_size = 20
    image_dim = 512
    hidden_dim = 8192
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 5
    max_epoch = 150

    bit = 64  # hash code length
    lr = 0.0001  # initial learning rate #0.00005
    stoimg_lr = 0.01
    stotxt_lr = 0.01
    logit_scale = 0.001
    device = 'cuda:0'

    # hyper-parameters10/0.1/10
    alpha = 10
    gamma = 10
    beta = 0.1
    mu = 0.00001
    lamb = 1

    margin = 0.4
    dropout = False
    stochastic_prior = 'uniform01'
    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 512
            self.training_size = 240 #k=1:240,k=2:480,4:960,8:1920,16:3840/7680
        if flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 512
            self.training_size = 210 #k=1:210,k=2:420,k=4:840,k=8:1680,k=16:3360
        if flag == 'coco':
            self.dataset = 'mscoco'
            self.db_size = 118285
            self.num_label = 80
            self.query_size = 5000
            self.text_dim = 512
            self.training_size = 6400#k_1=800,k=2:1600,k=4:3200,k=8:6400,k=16:12800

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
            
        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                    print('\t{0}: {1}'.format(k, getattr(self, k)))




opt = Default()
