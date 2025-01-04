import torch
from torch import nn
import os
from config import opt

class LinearCosRadius(nn.Module):
    '''
    Define the radius (R) as the linear function of the cos-similarity (t, v)
    '''
    def __init__(self, config: opt):
        super(LinearCosRadius, self).__init__()
        self.num_frames = 1
        self.embed_dim = 512

        self.linear_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.learnable_scalar = nn.Parameter(torch.Tensor(1))

        self._init_parameters()
        self.config = config

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, image_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            image_embeds: num_imgs x num_frames x embed_dim
        Output
            sims_out: num_imgs x embed_dim
        """
        # Normalize embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        # Assuming image_embeds is [batch_size, num_frames, embed_dim]
        # sims = torch.matmul(text_embeds.unsqueeze(1), image_embeds.permute(0, 2, 1))  # [batch_size, 1, num_frames]
        sims = torch.sum(text_embeds * image_embeds, dim=1, keepdim=True)
        # sims = sims.mean(dim=2)  # [batch_size, 1]

        # Linear projection
        sims_out = self.linear_proj(sims)  # [batch_size, embed_dim]

        return sims_out  # [batch_size, embed_dim]

class StochasticText(nn.Module):
    def __init__(self, config: opt):
        super(StochasticText, self).__init__()
        self.config = config
        self.std_branch = LinearCosRadius(config)

    def forward(self, text_features, image_features):
        """
        Input
            text_features: num_texts x embed_dim
            image_features: num_imgs x num_frames x embed_dim
        Output
            stochastic_text_features: num_texts x embed_dim
            text_mean: num_texts x embed_dim
            log_var: num_texts x embed_dim
        """
        # Mean is the original text features
        text_mean = text_features

        # Compute log variance using similarity-based radius
        log_var = self.std_branch(text_features, image_features)  # [batch_size, embed_dim]
        text_std = torch.exp(log_var)  # [batch_size, embed_dim]

        # Generate random noise based on the prior
        if self.config.stochastic_prior == 'uniform01':
            sigma = torch.rand_like(text_features)
        elif self.config.stochastic_prior == 'normal':
            sigma = torch.normal(mean=0., std=self.config.stochastic_prior_std, size=text_features.shape).to(text_std.device)
        else:
            raise NotImplementedError

        # Re-parameterization trick
        stochastic_text_features = text_mean + sigma * text_std

        return stochastic_text_features, text_mean, log_var

class StochasticImage(nn.Module):
    def __init__(self, config: opt):
        super(StochasticImage, self).__init__()
        self.config = config
        self.std_branch = LinearCosRadius(config)  # 可以共用同一个LinearCosRadius逻辑

    def forward(self, image_features, text_features):
        image_mean = image_features
        # 基于 image_features 和 text_features 计算 log_var
        log_var = self.std_branch(image_features, text_features)
        image_std = torch.exp(log_var)

        if self.config.stochastic_prior == 'uniform01':
            sigma = torch.rand_like(image_features)
        elif self.config.stochastic_prior == 'normal':
            sigma = torch.normal(mean=0., std=self.config.stochastic_prior_std, size=image_features.shape).to(image_std.device)
        else:
            raise NotImplementedError

        image_features_stochastic = image_mean + sigma * image_std
        return image_features_stochastic, image_mean, log_var
    
class LinearNet(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))

class GEN(torch.nn.Module):
    def __init__(self, dropout, image_dim=512, text_dim=512, hidden_dim=1024, output_dim=256):
        super(GEN, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim

        # self.config = config  # 确保 config 包含必要的参数，如 num_frames, embed_dim, stochastic_prior, stochastic_prior_std

        # Initialize T-MASS modules
        self.stochastic_text = StochasticText(opt)
        self.stochastic_image = StochasticText(opt)

        self.logit_scale = nn.Parameter(torch.ones([]))

        if dropout:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5)
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
        else:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True)
            )

        self.hash_module = nn.ModuleDict({
            'image': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
            'text': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
        })


    def forward(self, img, aug_img, txt, aug_txt, pool_txt, pool_img):

        image = torch.cat((img,aug_img), dim=0)
        text = torch.cat((txt,aug_txt), dim=0)
        pooled_txt = torch.cat((pool_txt,pool_txt), dim=0)
        pooled_img = torch.cat((pool_img,pool_img), dim=0)
        stochastic_text_features,  txt_mean, log_var_txt = self.stochastic_text(text, pooled_img)  # [batch_size, embed_dim]
        stochastic_img_features, img_mean, log_var_img = self.stochastic_image(image, pooled_txt)  # [batch_size, embed_dim]
        # liner_augimg = self.img_Net(aug_img)    # [batch_size, 512]
        # liner_augtxt = self.txt_Net(aug_txt)    # [batch_size, 512]

        # x = torch.cat((img,aug_img), dim=0)
        # y = torch.cat((txt,stochastic_text_features), dim=0)
        # x = torch.cat((img,liner_augimg), dim=0)
        # y = torch.cat((txt,liner_augtxt), dim=0)

        # f_x = self.image_module(image)
        f_x = self.image_module(stochastic_img_features)
        # f_y = self.text_module(text)
        f_y = self.text_module(stochastic_text_features)

        x_code = self.hash_module['image'](f_x).reshape(-1, self.output_dim)
        y_code = self.hash_module['text'](f_y).reshape(-1, self.output_dim)

        # return x_code, y_code, f_x, f_y, image, text
        return x_code, y_code, f_x, f_y, image, text, pooled_txt, pooled_img, stochastic_text_features, log_var_txt, stochastic_img_features, log_var_img

    def generate_img_code(self, i):
        
        f_i = self.image_module(i)

        code = self.hash_module['image'](f_i.detach()).reshape(-1, self.output_dim)
        return code
    
    def generate_img_code_val(self, i_query ,t_query):
        #测试中添加这一扰动部分
        i, _, _ = self.stochastic_text(t_query ,i_query) 
        f_i = self.image_module(i)

        code = self.hash_module['image'](f_i.detach()).reshape(-1, self.output_dim)
        return code

    def generate_txt_code(self, t):

        f_t = self.text_module(t)

        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.output_dim)
        return code
    
    def generate_txt_code_val(self, i_query ,t_query):
        
        t, _, _ = self.stochastic_text(t_query ,i_query)
        f_t = self.text_module(t)

        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.output_dim)
        return code
    
    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        else:
            torch.save(self.state_dict(), os.path.join(path, name))
        return name
