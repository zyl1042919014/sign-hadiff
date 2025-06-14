
from diffusion_net import *
from diffusion_util import *
import  torch
from Unet import Unet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SkelDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_steps):
        super().__init__()

        self.classifier_free = True
        self.n_poses = 1
        self.pose_dim = 150
        self.mlp_hidden_dim = 256
        self.hidden_dim = hidden_dim
        block_depth = 8
        self.in_size = 0
        self.num_steps = num_steps
        # if self.classifier_free:
        #     self.null_cond_prob = 1
        #     self.null_cond_emb = nn.Parameter(torch.randn(1, self.in_size))
        self.mlp = nn.Sequential(
            nn.Linear(self.pose_dim, self.mlp_hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.mlp_hidden_dim // 2, self.hidden_dim)
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.diffusion_net = DiffusionNet(
            # net=TransformerModel(num_pose=self.n_poses,
            #                      pose_dim=pose_dim,
            #                      embed_dim=pose_dim + 3 + self.in_size,
            #                      hidden_dim=diff_hidden_dim,
            #                      depth=block_depth // 2,
            #                      decoder_depth=block_depth // 2
            #                      ),
            net=Unet(self.hidden_dim),
            var_sched=VarianceSchedule(
                num_steps=self.num_steps,
                beta_1=1e-4,
                beta_T=0.02,
                mode='linear'
            )
        )

    def get_loss(self, x):
        x_feature = self.norm(self.mlp(x))
        neg_elbo = self.diffusion_net.get_loss(x, x_feature)
        return neg_elbo

    def get_loss_for_united_train(self, x, x_feature):
        # 这里输入的x和x_feature都是(batch, seq_len, dim)的格式
        # 需要改成(batch * seq_len, 1, dim)的格式
        batch_size = x.size(0)
        seq_len = x.size(1)
        dim1 = x.size(2)
        dim2 = x_feature.size(2)
        x_for_diffusion = x.view(batch_size * seq_len, dim1).unsqueeze(1)
        x_feature_for_diffusion = x_feature.view(batch_size * seq_len, dim2).unsqueeze(1)
        loss = self.diffusion_net.get_loss(x_for_diffusion, x_feature_for_diffusion)
        return loss

    def get_hidden_space(self, x):
        return self.norm(self.mlp(x))

    # def sample(self, pose_dim, in_audio, pre_seq, reference):
    #
    #     if self.input_context == 'audio':
    #         audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)
    #         in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
    #         # in_data = in_audio
    #
    #     if self.classifier_free:
    #         uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1], 1).unsqueeze(0)
    #         samples = self.diffusion_net.sample(self.n_poses, in_data, pose_dim, reference,
    #                                             uncondition_embedding=uncondition_embedding)
    #     else:
    #         samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, reference)
    #     return samples
    def sample(self, x):
        x = self.get_hidden_space(x)

        x = self.norm(x)

        # x[:10, :, :] += 0.1
        # x[11:, :, :] += 0.3
        # len_frame = x.shape[0]
        # for i in range(len_frame):
        #     tmp = torch.randn(1).to(device)
        #     x[i, :, :] += tmp / 100
        samples = self.diffusion_net.sample(x)
        return samples

    def sample_after_mlp(self, x):
        samples = self.diffusion_net.sample(x)
        return samples

    def freeze_mlp(self):
        # 冻结MLP中的线性层
        for idx, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                # 冻结线性层的参数
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        self.norm.weight.requires_grad = False
        self.norm.bias.requires_grad = False

    def freeze_unet(self):
        for param in self.diffusion_net.parameters():
            param.requires_grad = False
