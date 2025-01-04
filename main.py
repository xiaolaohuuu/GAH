import os
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import *
from config import opt
from models.dis_model import DIS
from dadh_ours.models.GAH_model import GEN
from torch.optim import Adam
from utils import *
import time
import pickle
import random
import argparse

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

def train(**kwargs):
    opt.parse(kwargs)
    opt.beta = opt.beta + 0.1

    dset = load_coco(mode='train')

    train_data = my_traindataset(dset.img_feature, dset.img_augfeature, dset.txt_feature, dset.txt_augfeature, dset.pooled_average_txtfeature, dset.pooled_average_imgfeature,dset.label)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    L = dset.label.cuda()

    dset_query = load_coco(mode='query')
    i_query_data = dset_query.img_feature
    t_query_data = dset_query.txt_feature
    query_labels = torch.from_numpy(dset_query.label).float().cuda()

    dset_retrieval = load_coco(mode='retrieval')
    t_db_data = dset_retrieval.txt_feature
    i_db_data = dset_retrieval.img_feature
    db_labels = torch.from_numpy(dset_retrieval.label).float().cuda()

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False)

    generator = GEN(opt.dropout, opt.image_dim, opt.text_dim, opt.hidden_dim, opt.bit).cuda()

    discriminator = DIS(opt.hidden_dim//4, opt.hidden_dim//8, opt.bit).cuda()

    optimizer = Adam([
        {'params': generator.image_module.parameters()},
        {'params': generator.text_module.parameters()},
        {'params': generator.hash_module.parameters()},
        {'params': generator.stochastic_text.parameters(), 'lr': opt.stotxt_lr},
        {'params': generator.stochastic_image.parameters(), 'lr': opt.stoimg_lr},
        {'params': generator.logit_scale, 'lr': opt.logit_scale}
    ], lr=opt.lr, weight_decay=0.0005)  

    optimizer_dis = {
        'feature': Adam(discriminator.feature_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001),
        'hash': Adam(discriminator.hash_dis.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)
    }

    tri_loss = TripletLoss(opt, reduction='sum')

    loss = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = [] 
    train_times = []

    B_i = torch.randn(opt.training_size, opt.bit).sign().cuda()
    B_t = B_i
    H_i = torch.zeros(opt.training_size, opt.bit).cuda()
    H_t = torch.zeros(opt.training_size, opt.bit).cuda()
    loss_fn = CLIPLoss()
    for epoch in range(opt.max_epoch):
        t1 = time.time()
        e_loss = 0
        for i, (ind, img, aug_img, txt, aug_txt, pool_txt, pool_img, label) in tqdm(enumerate(train_dataloader)):
            img = img.cuda()
            txt = txt.cuda()
            aug_img = aug_img.cuda()
            aug_txt = aug_txt.cuda()
            labels = label.cuda()
            pool_txt = pool_txt.cuda()
            pool_img = pool_img.cuda()

            ind = torch.cat((ind,ind), dim=0)#TODO:
            labels = torch.cat((labels,labels), dim=0)

            batch_size = len(ind)

            h_i, h_t, f_i, f_t, image, text, pooled_txt, pooled_img,stochastic_text_features, log_var_txt, stochastic_img_features, log_var_img = generator(img, aug_img, txt, aug_txt, pool_txt, pool_img)


            sims_txt = sim_matrix_training(stochastic_text_features, pooled_img, pooling_type='avg')  
            sims_img = sim_matrix_training(stochastic_img_features, pooled_txt, pooling_type='avg')  


            loss_base = loss_fn(sims_txt, generator.logit_scale)
            loss_base_img = loss_fn(sims_img, generator.logit_scale)


            image_embeds_avg = pooled_img 
            pointer = image_embeds_avg - text 
            text_support = pointer / pointer.norm(dim=-1, keepdim=True) * torch.exp(log_var_txt) + text 
            output_support = sim_matrix_training(text_support, image, pooling_type='avg')  
            loss_support = loss_fn(output_support, generator.logit_scale) 

            # image support
            text_embeds_avg = pooled_txt  # point
            pointer_image = text_embeds_avg - image  
            image_support = pointer_image / pointer_image.norm(dim=-1, keepdim=True) * torch.exp(log_var_img) + image
            # image_support  text calsimilarity
            output_support_image = sim_matrix_training(text, image_support, pooling_type='avg')
            loss_support_image = loss_fn(output_support_image, generator.logit_scale)

            H_i[ind, :] = h_i.data
            H_t[ind, :] = h_t.data
            h_t_detach = generator.generate_txt_code(text)

            #####
            # train feature discriminator
            #####
            D_real_feature = discriminator.dis_feature(f_i.detach())
            D_real_feature = -opt.gamma * torch.log(torch.sigmoid(D_real_feature)).mean()
            # D_real_feature = -D_real_feature.mean()
            optimizer_dis['feature'].zero_grad()
            D_real_feature.backward()

            # train with fake
            D_fake_feature = discriminator.dis_feature(f_t.detach())
            D_fake_feature = -opt.gamma * torch.log(torch.ones(batch_size).cuda() - torch.sigmoid(D_fake_feature)).mean()
            # D_fake_feature = D_fake_feature.mean()
            D_fake_feature.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.hidden_dim//4).cuda()
            interpolates = alpha * f_i.detach() + (1 - alpha) * f_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_feature(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            # 10 is gradient penalty hyperparameter
            feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            feature_gradient_penalty.backward()

            optimizer_dis['feature'].step()

            #####
            # train hash discriminator
            #####
            D_real_hash = discriminator.dis_hash(h_i.detach())
            D_real_hash = -opt.gamma * torch.log(torch.sigmoid(D_real_hash)).mean()
            optimizer_dis['hash'].zero_grad()
            D_real_hash.backward()

            # train with fake
            D_fake_hash = discriminator.dis_hash(h_t.detach())
            D_fake_hash = -opt.gamma * torch.log(torch.ones(batch_size).cuda() - torch.sigmoid(D_fake_hash)).mean()
            D_fake_hash.backward()

            # train with gradient penalty
            alpha = torch.rand(batch_size, opt.bit).cuda()
            interpolates = alpha * h_i.detach() + (1 - alpha) * h_t.detach()
            interpolates.requires_grad_()
            disc_interpolates = discriminator.dis_hash(interpolates)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)

            hash_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            hash_gradient_penalty.backward()

            optimizer_dis['hash'].step()

            loss_G_txt_feature = -torch.log(torch.sigmoid(discriminator.dis_feature(f_t))).mean()
            loss_adver_feature = loss_G_txt_feature 

            loss_G_txt_hash = -torch.log(torch.sigmoid(discriminator.dis_hash(h_t_detach))).mean()
            loss_adver_hash = loss_G_txt_hash 

            tri_i2t = tri_loss(h_i, labels, target=h_t, margin=opt.margin)
            tri_t2i = tri_loss(h_t, labels, target=h_i, margin=opt.margin)
            weighted_cos_tri = tri_i2t + tri_t2i 

            err = opt.alpha * weighted_cos_tri + \
                opt.gamma * (loss_adver_feature + loss_adver_hash) + opt.beta * (loss_support + loss_base + loss_base_img + loss_support_image)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            e_loss = err + e_loss

        P_i = torch.inverse(
                L.t() @ L + opt.lamb * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B_i
        B_i = (L @ P_i + 0.5 * opt.mu * (H_i + H_t)).sign()
        loss.append(e_loss.item())
        print('...epoch: %3d, loss: %3.3f' % (epoch + 1, loss[-1]))
        delta_t = time.time() - t1


        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                                   query_labels, db_labels)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(generator)

        if epoch % 100 == 0:
            for params in optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.8, 1e-6)

    if not opt.valid:
        save_model(generator)

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(generator, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                               query_labels, db_labels)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    with open(os.path.join(path, 'result.pkl'), 'wb') as f:
        pickle.dump([train_times, mapi2t_list, mapt2i_list], f)


def valid(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels):
    model.eval()

    qBX = generate_img_code(model, x_query_dataloader, opt.query_size)
    qBY = generate_txt_code(model, y_query_dataloader, opt.query_size)
    rBX = generate_img_code(model, x_db_dataloader, opt.db_size)
    rBY = generate_txt_code(model, y_db_dataloader, opt.db_size)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    model.train()
    return mapi2t.item(), mapt2i.item()

def generate_img_code(model, i_dataloader, num):
    B = torch.zeros(num, opt.bit).cuda()

    for i, input_data in tqdm(enumerate(i_dataloader)):
        input_data = input_data.cuda()
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, t_dataloader, num):
    B = torch.zeros(num, opt.bit).cuda()

    for i, input_data in tqdm(enumerate(t_dataloader)):
        input_data = input_data.cuda()
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))

def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', type=int, required=True, help='Bit value')
    parser.add_argument('--flag', type=str, required=True, help='Bit value')
    args = parser.parse_args()
    set_seed(2024)
    train(flag=args.flag, bit=args.bit)
