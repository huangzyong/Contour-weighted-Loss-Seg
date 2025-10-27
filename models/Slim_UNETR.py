import torch
import torch.nn as nn
from thop import profile


class PatchPartition(nn.Module):
    def __init__(self, channels):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        return x


class LineConv(nn.Module):
    def __init__(self, channels):
        super(LineConv, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(
            channels, channels * expansion, kernel_size=1, bias=False
        )
        self.act = nn.GELU()
        self.line_conv_1 = nn.Conv3d(
            channels * expansion, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x


class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.pointwise_conv_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(
            channels, channels, padding=1, kernel_size=3, groups=channels, bias=False
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.pointwise_conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose3d(
            channels, channels, kernel_size=r, stride=r, groups=channels
        )
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class Block(nn.Module):
    def __init__(self, channels, r, heads):
        super(Block, self).__init__()

        self.patch1 = PatchPartition(channels)
        self.LocalRC = LocalRepresentationsCongregation(channels)
        self.LineConv1 = LineConv(channels)
        self.patch2 = PatchPartition(channels)
        self.GlobalST = GlobalSparseTransformer(channels, r, heads)
        self.LocalRD = LocalReverseDiffusion(channels, r)
        self.LineConv2 = LineConv(channels)

    def forward(self, x):
        x = self.patch1(x) + x
        x = self.LocalRC(x) + x
        x = self.LineConv1(x) + x
        x = self.patch2(x) + x
        x = self.LocalRD(self.GlobalST(x)) + x
        x = self.LineConv2(x) + x
        return x

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=384,
        embedding_dim=27,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(Encoder, self).__init__()
        self.DWconv1 = DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[0], r=4)
        self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=2)
        self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=2)
        self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=2)
        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, embedding_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        B, C, W, H, Z = x.shape
        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, hidden_states_out, (B, C, W, H, Z)

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(Decoder, self).__init__()
        self.SegHead = TransposedConvLayer(
            dim_in=channels[0], dim_out=out_channels, r=4
        )
        self.TSconv3 = TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=2)
        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=2)
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=2)

        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

    def forward(self, x, hidden_states_out, x_shape):
        B, C, W, H, Z = x_shape
        x = x.reshape(B, C, W, H, Z)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + hidden_states_out[2]
        x = self.block3(x)
        x = self.TSconv2(x)
        x = x + hidden_states_out[1]
        x = self.block2(x)
        x = self.TSconv3(x)
        x = x + hidden_states_out[0]
        x = self.block1(x)
        x = self.SegHead(x)
        return x

class SlimUNETR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: deepest semantic channels
            embedding_dim: position code length
            channels: selection list of downsampling feature channel
            blocks: depth list of slim blocks
            heads: multiple set list of attention computations in parallel
            r: list of stride rate
            dropout: dropout rate
        Examples::
            # for 3D single channel input with size (128, 128, 128), 3-channel output.
            >>> net = SlimUNETR(in_channels=4, out_channels=3, embedding_dim=64)

            # for 3D single channel input with size (96, 96, 96), 2-channel output.
            >>> net = SlimUNETR(in_channels=1, out_channels=2, embedding_dim=27)

        """
        super(SlimUNETR, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

    def forward(self, x):
        embeding, hidden_states_out, (B, C, W, H, Z) = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out, (B, C, W, H, Z))
        return x


if __name__ == "__main__":
    x = torch.randn(size=(1, 4, 128, 160, 160))
    model = SlimUNETR(
        in_channels=4,
        out_channels=4,
        embed_dim=96,
        embedding_dim=100,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        # distillation=False,
        dropout=0.3,
    )
    print(model(x).shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    flops, params = profile(model, inputs=(x, ))
    print(flops, params)