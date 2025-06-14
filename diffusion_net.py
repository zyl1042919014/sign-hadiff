import math
import torch
import torch.nn.functional as F
import tqdm
from torch.nn import Module, MSELoss
from diffusion_util import VarianceSchedule, TransformerModel
from Unet import Unet


class DiffusionNet(Module):

    # def __init__(self, net: TransformerModel, var_sched: VarianceSchedule):
    #     super().__init__()
    #     self.net = net
    #     self.var_sched = var_sched
    def __init__(self, net: Unet, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):

        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim),
                          reduction='mean')
        # loss = total_loss(groundTruth=e_theta.contiguous().view(-1, point_dim),
        #                   prediction=e_rand.contiguous().view(-1, point_dim),
        #                   loss_function=MSELoss(), scale_wrist=0.1, scale_body=0.3, scale_hand=0.6)

        return loss

    def get_loss_one_frame(self, target_skel, t=None):

        batch_size, _, point_dim = target_skel.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(target_skel)  # (B, N, d)
        e_theta = self.net(c0 * target_skel + c1 * e_rand, beta=beta)

        loss = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim),
                          reduction='mean')

        return loss

    def sample(self, context, flexibility=0.0, ret_traj=False):
        num_pose = 1
        pose_dim = 150
        batch_size = context.size(0)
        # torch.manual_seed(27)
        # torch.cuda.manual_seed_all(27)
        x_T = torch.randn([batch_size, num_pose, pose_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in tqdm.tqdm(range(self.var_sched.num_steps, 0, -1)):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]

            e_theta = self.net(x_t, beta=beta, context=context)

            t0 = 25
            if t0 > t > 1:
                sigma_a = 1 / t0 / t0 * (t - t0) * (t - t0)
                z0 = sigma_a * torch.randn_like(x_T[:, 0, :].unsqueeze(1))

                res = torch.zeros_like(z)
                for n in range(num_pose):
                    zn = math.sqrt((1 - sigma_a * sigma_a)) * torch.randn_like(x_T[:, 0, :].unsqueeze(1))
                    res[:, n:n + 1, :] = zn + z0
                z = res

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()
            # difference = torch.sum(traj[t-1] - reference)
            # print(difference)
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
