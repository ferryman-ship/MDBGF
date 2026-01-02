import torch
import torch.nn as nn


class BaseGenerator(nn.Module):
    def __init__(
        self,
        noise_dim,
        structure_dim,
        img_dim
    ):
        super(BaseGenerator, self).__init__()
        self.proj_dim = 512
        self.noise_dim = noise_dim
        self.generator_model = nn.Sequential(
            nn.Linear(noise_dim + structure_dim, self.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.proj_dim, img_dim)
        )


    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        batch_data = torch.cat((random_noise, batch_ent_emb), dim=-1)
        out = self.generator_model(batch_data) # 320
        return out

class MultiGenerator(nn.Module):
    def __init__(
            self,
            noise_dim,
            structure_dim,
            img_dim
    ):
        super(MultiGenerator, self).__init__()
        self.img_generator = BaseGenerator(noise_dim, structure_dim, img_dim)
        self.text_generator = BaseGenerator(noise_dim, structure_dim, img_dim)

    def forward(self, batch_ent_emb, modal):
        if modal == 1:
            return self.img_generator(batch_ent_emb)
        elif modal == 2:
            return self.text_generator(batch_ent_emb)
        else:
            raise NotImplementedError