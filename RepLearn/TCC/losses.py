
import torch
from torch.nn import functional as F
from scipy.spatial.distance import cdist

from RepLearn.TCC.utils import get_lav_weights, generate_unique_video_steps
from VAOT import asot, CIDM_regularization


def calculate_similarity(embeddings1, embeddings2, temperature):
    nc = embeddings1.size(1)
    # L2 distance
    emb1_norm = (embeddings1**2).sum(dim=1)
    emb2_norm = (embeddings2**2).sum(dim=1)
    # Trick used to calculate the distance matrix without any loops. It uses
    # the fact that (a - b)^2 = a^2 + b^2 - 2ab.
    dist = torch.max(
        emb1_norm + emb2_norm - 2.0 * torch.matmul(
            embeddings1,
            embeddings2.t()
        ),
        torch.tensor(
            0.0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    # similarity: (N, M)
    similarity = -1.0 * dist
    similarity /= embeddings1.size(1)
    similarity /= temperature
    return similarity


def embeddings_similarity(embeddings1, embeddings2, temperature):
    '''
    embeddings1: (N, D). in paper, U. u_i, i=1 to N
    embeddings2: (M, D). in paper, V. v_j, j=1 to M
    '''
    max_num_frames = embeddings1.size(0)

    similarity = calculate_similarity(embeddings1, embeddings2, temperature)
    similarity = F.softmax(similarity, dim=1)
    # v_tilda
    soft_nearest_neighbor = torch.matmul(similarity, embeddings2)

    # logits for Beta_k
    logits = calculate_similarity(
        soft_nearest_neighbor,
        embeddings1,
        temperature
    )
    # labels = F.one_hot(
    #     torch.tensor(range(max_num_frames)),
    #     num_classes=max_num_frames
    # )
    labels = torch.eye(max_num_frames)[torch.tensor(range(max_num_frames))]

    return logits, labels


def contrastive_idm_loss(
    embeddings,
    steps,
    steps_norm,
    lambda_=2.0,
    sigma=15.0
):
    lambda_ = torch.tensor(lambda_, requires_grad=True).cuda()
    sigma = torch.tensor(sigma, requires_grad=True).cuda()
    unique_vid_steps = generate_unique_video_steps(embeddings, steps)
    unique_vid_steps_norm = generate_unique_video_steps(embeddings, steps_norm)
    assert embeddings.shape[0] == len(unique_vid_steps)
    losses = torch.tensor(()).to('cuda')
    for video_count, (single_video_embds, video_steps) in enumerate(
        zip(embeddings, unique_vid_steps_norm)
    ):
        loss = torch.tensor([0.0]).to('cuda')
        w, w_dash = get_lav_weights(video_steps)
        w = torch.tensor(w, requires_grad=True).to('cuda')
        w_dash = torch.tensor(w_dash, requires_grad=True).to('cuda')
        temporal_dist = torch.abs(
            unique_vid_steps[video_count].unsqueeze(dim=0) - \
                unique_vid_steps[video_count].unsqueeze(dim=1)
        )
        y_ = (
            torch.ones(temporal_dist.shape, requires_grad=True).cuda() * \
                (temporal_dist.cuda() > sigma.to(torch.int64)).to(torch.float32)
        )
        ## Calculating the self-distance matrix
        dist_calculation = single_video_embds.detach().cpu().numpy()
        self_dist_D = torch.tensor(cdist(
            dist_calculation,
            dist_calculation,
        )).to('cuda').to(torch.float32)
        max_values = torch.max(
            torch.tensor(0, dtype=torch.float32).to('cuda'),
            lambda_ - self_dist_D
        )
        loss = y_ * w_dash.to(torch.float32) * max_values + (torch.tensor(1.0).cuda() - y_) * w.to(torch.float32) * self_dist_D
        losses = torch.cat((losses, loss.sum().unsqueeze(dim=0)), 0)
    return losses.mean()


def cycleback_regression_loss(
    logits,
    labels,
    num_frames,
    steps,
    seq_lens,
    normalize_indices,
    variance_lambda,
):
    labels = labels.detach().cuda()  # (bs, ts)
    steps = steps.detach().cuda()  # (bs, ts)
    steps = steps.float().cuda()
    seq_lens = seq_lens.float().cuda()

    seq_lens = seq_lens.unsqueeze(1).repeat(1, num_frames).cuda()
    steps = steps / seq_lens

    # After using torch.nn.DataParallel, logits are on 'cuda' and rest of the
    # things are on 'cpu'. Moving beta to 'cuda' fixes the issue.
    beta = F.softmax(logits, dim=1).to('cuda')
    true_timesteps = (labels * steps).sum(dim=1)
    pred_timesteps = (beta * steps).sum(dim=1)
    pred_timesteps_repeated = pred_timesteps.unsqueeze(1).repeat(1, num_frames)
    pred_timesteps_var = (
        (steps - pred_timesteps_repeated)**2 * beta
    ).sum(dim=1)
    pred_timesteps_log_var = pred_timesteps_var.log()
    squared_error = (true_timesteps - pred_timesteps)**2
    loss = torch.mean(
        (-pred_timesteps_log_var).exp() * squared_error + variance_lambda * \
            pred_timesteps_log_var
    )
    return loss


def temporal_cycle_consistency_loss(
    embeddings,
    steps,
    seq_lens,
    cfg,
    num_frames,
    batch_size,
    temperature,
    variance_lambda,
    normalize_indices,
    writer=None,
    iter_count=None
):
    logits_list = []
    labels_list = []
    steps_list = []
    seq_lens_list = []
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                logits, labels = embeddings_similarity(
                    embeddings[i],
                    embeddings[j],
                    temperature
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i:i+1].repeat(num_frames, 1))
                seq_lens_list.append(seq_lens[i:i+1].repeat(num_frames))
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    loss = cycleback_regression_loss(
        logits,
        labels,
        num_frames,
        steps,
        seq_lens,
        normalize_indices,
        variance_lambda
    )

    if cfg.LAV.USE_CIDM:
        contrastive_loss = contrastive_idm_loss(
            embeddings,
            steps,
            steps / seq_lens.unsqueeze(1).repeat(1, num_frames),
            lambda_ = cfg.LAV.LAMBDA,
            sigma=cfg.LAV.SIGMA
        )
        if writer is not None:
            writer.add_scalar(
                'Loss/C-IDM',
                (cfg.LAV.CONTRIB_PERCENT * contrastive_loss).item(),
                iter_count
            )
            writer.add_scalar(
                'Loss/TCC',
                loss.item(),
                iter_count
            )
        return cfg.LAV.CONTRIB_PERCENT * contrastive_loss + loss

    return loss

def vaot_loss(embs, config):
    # This function MUST recieve embs of shape (batch_size=2, num_main_frames, embedding_size)
    features_X, features_Y = torch.split(embs, 1, dim=0)

    T_X = features_X.shape[1]
    T_Y = features_Y.shape[1]
    # Eq (6)
    # codes represent a matrix P for each batch element
    # size of a matrix P is (no. of frames in X x no. of frames in Y)
    # P_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
    codes = torch.exp(features_X @ features_Y.transpose(1, 2) / config.VAOT.temp)
    codes = codes / codes.sum(dim=-1, keepdim=True)

    # Produce pseudo-labels using ASOT, note that we don't backpropagate through this part
    with torch.no_grad():
        # Calculate the KOT cost matrix from the paragraph above Eq (7)
        # ρR = rho * Temporal prior
        temp_prior = asot.temporal_prior(T_X, T_Y, config.VAOT.rho, features_X.device)
        # Cost Matrix Ck from section 4.2, no need to divide by norms since both vectors were previously normalized with F.normalize()
        cost_matrix = 1. - features_X @ features_Y.transpose(1, 2)
        # Ĉk = Ck + ρR
        cost_matrix += temp_prior


        ## Added for virtual frames
        B, N, K = cost_matrix.shape
        dev = cost_matrix.device
        top_row = torch.ones(B, 1, K).to(dev) * config.VAOT.zeta
        cost_matrix = torch.cat((top_row, cost_matrix), dim=1)
        left_column = torch.ones(B, N + 1, 1).to(dev) * config.VAOT.zeta
        cost_matrix = torch.cat((left_column, cost_matrix), dim=2)


        # opt_codes represent a matrix Tb for each batch element
        # size of a matrix Tb is (no. of frames in X x no. of frames in Y)
        # Tb are the (soft) pseudo-labels defined above Eq (7)
        # Tb_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
        opt_codes, _ = asot.segment_asot(cost_matrix=cost_matrix,
                                            eps=config.VAOT.train_eps, alpha=config.VAOT.alpha_train, radius=config.VAOT.radius_gw,
                                            ub_frames=config.VAOT.ub_frames, ub_actions=config.VAOT.ub_actions,
                                            lambda_frames=config.VAOT.lambda_frames_train,
                                            lambda_actions=config.VAOT.lambda_actions_train,
                                            n_iters=config.VAOT.n_ot_train, step_size=config.VAOT.step_size)

    # Eq (7)
    # Modified version that doesn't use mask_X and mask_Y
    loss_ce = -((opt_codes * torch.log(codes + config.VAOT.num_eps))).sum(dim=2).mean()
    return loss_ce

def vaot_loss_with_CIDM(embs, steps, seq_lens, config):
    # This function MUST recieve embs of shape (batch_size=2, num_main_frames, embedding_size)
    features_X, features_Y = torch.split(embs, 1, dim=0)

    T_X = features_X.shape[1]
    T_Y = features_Y.shape[1]
    # Eq (6)
    # codes represent a matrix P for each batch element
    # size of a matrix P is (no. of frames in X x no. of frames in Y)
    # P_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
    codes = torch.exp(features_X @ features_Y.transpose(1, 2) / config.VAOT.temp)
    codes = codes / codes.sum(dim=-1, keepdim=True)

    # Produce pseudo-labels using ASOT, note that we don't backpropagate through this part
    with torch.no_grad():
        # Calculate the KOT cost matrix from the paragraph above Eq (7)
        # ρR = rho * Temporal prior
        temp_prior = asot.temporal_prior(T_X, T_Y, config.VAOT.rho, features_X.device)
        # Cost Matrix Ck from section 4.2, no need to divide by norms since both vectors were previously normalized with F.normalize()
        cost_matrix = 1. - features_X @ features_Y.transpose(1, 2)
        # Ĉk = Ck + ρR
        cost_matrix += temp_prior


        ## Added for virtual frames
        B, N, K = cost_matrix.shape
        dev = cost_matrix.device
        top_row = torch.ones(B, 1, K).to(dev) * config.VAOT.zeta
        cost_matrix = torch.cat((top_row, cost_matrix), dim=1)
        left_column = torch.ones(B, N + 1, 1).to(dev) * config.VAOT.zeta
        cost_matrix = torch.cat((left_column, cost_matrix), dim=2)


        # opt_codes represent a matrix Tb for each batch element
        # size of a matrix Tb is (no. of frames in X x no. of frames in Y)
        # Tb are the (soft) pseudo-labels defined above Eq (7)
        # Tb_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
        opt_codes, _ = asot.segment_asot(cost_matrix=cost_matrix,
                                            eps=config.VAOT.train_eps, alpha=config.VAOT.alpha_train, radius=config.VAOT.radius_gw,
                                            ub_frames=config.VAOT.ub_frames, ub_actions=config.VAOT.ub_actions,
                                            lambda_frames=config.VAOT.lambda_frames_train,
                                            lambda_actions=config.VAOT.lambda_actions_train,
                                            n_iters=config.VAOT.n_ot_train, step_size=config.VAOT.step_size)

    # Eq (7)
    # Modified version that doesn't use mask_X and mask_Y
    loss_ce = -((opt_codes * torch.log(codes + config.VAOT.num_eps))).sum(dim=2).mean()

    ############################################################################ CIDM START
    a_embs, b_embs = features_X, features_Y
    a_steps, b_steps = torch.split(steps, 1, dim=0)
    a_seq_len, b_seq_len = [x.to(torch.float64) for x in torch.split(seq_lens, 1, dim=0)]
    # Normalizing steps to make the input look like LAV's, since we use the LAV implementation of C-IDM
    a_steps = a_steps.to(torch.float64) / a_seq_len.unsqueeze(1)
    b_steps = b_steps.to(torch.float64) / b_seq_len.unsqueeze(1)

    # BUG sigma might need to change for diff datasets, check config comments
    CIDM = CIDM_regularization.Contrastive_IDM(sigma=config.LAV.SIGMA, margin=config.LAV.LAMBDA)

    cidm_regularization_term = CIDM(
        a_embs,           # shape: [1, VAOT.NUM_FRAMES=32, VAOT.EMBEDDING_SIZE=128]
        b_embs,           # shape: [1, VAOT.NUM_FRAMES=32, VAOT.EMBEDDING_SIZE=128]
        a_steps[0],       # shape: [VAOT.NUM_FRAMES=32]
        b_steps[0],       # shape: [VAOT.NUM_FRAMES=32]
        a_seq_len[0],     # scalar
        b_seq_len[0]      # scalar
    )
    ############################################################################ CIDM END

    # alpha is the regularization weight from LAV eq 8
    total_loss = loss_ce + (config.LAV.ALPHA * cidm_regularization_term)
    
    total_loss = total_loss / config.VAOT.NUM_FRAMES

    return total_loss
