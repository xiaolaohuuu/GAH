import torch
import numpy as np
import visdom


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)
    
from torch import nn
import torch
from torch.nn import functional as F


def cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
    distances = torch.clamp(1 - cos_sim, 0)

    return distances


def get_triplet_mask(s_labels, t_labels, opt):
    flag = (opt.beta - 0.1) * opt.gamma
    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).to(opt.device)
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z

    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = (sim_pos - sim_neg) * (flag + 0.1)
    mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

    return mask, weight


class TripletLoss(nn.Module):
    def __init__(self, opt, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.opt = opt

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist = cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = get_triplet_mask(s_labels, t_labels, self.opt)
        if self.opt.alpha == 10:
            triplet_loss = 10 * weight * mask * triplet_loss
        else:
            triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss.clamp(0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss


import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
import gc

def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        raise NotImplementedError("Only 'avg' pooling_type is supported.")
        # # num_texts x embed_dim x num_vids
        # vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # # num_texts x 1 x embed_dim
        # text_embeds = text_embeds.unsqueeze(1)
        
        # sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference_stochastic(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):

    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1,
                                                                                                          keepdim=True)

    if pooling_type == 'avg':
        print(f'for this case, have not tried')
        raise NotImplementedError

    else:
        num_txts, num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1, 2, 3, 0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.reshape(num_vids * max_text_per_vid, embed_dim,
                                                                                num_vids)
        text_embeds_per_video_id = text_embeds_per_video_id.permute(0, 2, 1, 3)
        text_embeds_per_video_id = text_embeds_per_video_id.reshape(num_vids * max_text_per_vid, num_txts, embed_dim)


        sims = torch.bmm(text_embeds_per_video_id,
                         vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, num_txts, num_vids)
        sims_diag = torch.stack([sims[i, :, :, i] for i in range(sims.shape[0])],
                                dim=-1)
        print(f'>>>check sims_diag={sims_diag.shape}')
        sims_diag = sims_diag.permute(1, 0, 2)

    return sims_diag


def sim_matrix_inference_stochastic_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type,
                                                 batch_size_split, config):

    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)



    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1,
                                                                                                          keepdim=True)

    if pooling_type == 'avg':

        print(f'for this case, have not tried')
        raise NotImplementedError

    else:
        num_vids, num_txts, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1, 2, 3, 0)



        text_embeds_per_video_id = text_embeds_per_video_id.permute(0, 2, 1, 3)

        msg = (f'>>>text_embeds_per_video_id={text_embeds_per_video_id.shape}')


        batch_size = text_embeds_per_video_id.shape[0]
        if batch_size_split is None:
            batch_size_split = 1
        else:
            pass


        dim0, dim1, dim2, dim3 = text_embeds_per_video_id.shape
        sims_diag = torch.zeros(dim1, dim0, dim2)

        for batch in range(0, batch_size, batch_size_split):
            tensor1_batch = text_embeds_per_video_id[batch: min(batch + batch_size_split, batch_size)]

            tensor2_batch = vid_embeds_pooled_per_video_id[batch: min(batch + batch_size_split, batch_size)]


            result_batch = torch.matmul(tensor1_batch, tensor2_batch)
            msg = (f'batch={batch} result_batch={result_batch.shape}')


            for idx in range(batch, min(batch + batch_size_split, batch_size)):
                sims_diag[:, :, idx] = result_batch[idx - batch, :, :,
                                       idx]

        del text_embeds_per_video_id, vid_embeds_pooled_per_video_id
        gc.collect()


        msg = (f'>>>check sims_diag={sims_diag.shape}')


        sims_diag = sims_diag.permute(1, 0, 2)

    return sims_diag


def generate_embeds_per_video_id_stochastic(text_embeds_stochastic_allpairs, vid_embeds_pooled, all_vid_ids,
                                            pooling_type):
    # Construct dictionary of text embeds per unique video id
    if pooling_type == 'avg':
        # num_vids x embed_dim
        text_embeds_per_video_id = text_embeds_stochastic_allpairs

    else:
        # Construct dictionary of video embeds for each text per video_id
        text_embeds_per_video_id = []

        for i in range(text_embeds_stochastic_allpairs.shape[0]):
            text_embeds_per_video_id.append({})
            for idx, t_id in enumerate(all_vid_ids):
                if t_id in text_embeds_per_video_id[i]:
                    text_embeds_per_video_id[i][t_id].append(text_embeds_stochastic_allpairs[i, idx, :])
                else:
                    text_embeds_per_video_id[i][t_id] = [text_embeds_stochastic_allpairs[i, idx, :]]

        for i in range(len(text_embeds_per_video_id)):
            for t_id in text_embeds_per_video_id[i]:
                text_embeds_per_video_id[i][t_id] = torch.stack(text_embeds_per_video_id[i][t_id])

            text_embeds_per_video_id[i] = pad_and_stack_dict_to_tensor(text_embeds_per_video_id[i],
                                                                       text_embeds_per_video_id[i].keys(),
                                                                       text_embeds_stochastic_allpairs.shape[-1])

        text_embeds_per_video_id = torch.stack(text_embeds_per_video_id)

    if pooling_type == 'avg':
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                                                                             vid_embeds_pooled_per_video_id[i].keys(),
                                                                             vid_embeds_pooled.shape[-1])

        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1,0,2)
    
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims!=sims] = float('-inf')
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy() # diagonal

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input
