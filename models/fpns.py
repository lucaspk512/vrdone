from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, LayerNorm

class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln)
            )
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.scale_factor, mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(used_backbone_levels):
            x, new_mask = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )
            new_fpn_masks += (new_mask, )

        return fpn_feats, new_fpn_masks

class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = #levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks


class FPN1D_Fuse(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
        norm_first=False,
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.input_norms = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.lateral_norms = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            if i == self.end_level - 1:
                l_conv = None
                # use depthwise conv here for efficiency
                fpn_conv = MaskedConv1D(
                    in_channels[i], out_channel, 3,
                    padding=1, bias=(not with_ln), groups=out_channel
                )

                # layer norm for order (B C T)
                if with_ln:
                    if norm_first:
                        input_norm = LayerNorm(in_channels[i])
                    else:
                        input_norm = nn.Identity()

                    fpn_norm = LayerNorm(out_channel)
                else:
                    fpn_norm = nn.Identity()

                lateral_norm = None
            else:
                l_conv = MaskedConv1D(
                    in_channels[i], out_channel, 1, bias=(not with_ln)
                )
                # use depthwise conv here for efficiency
                fpn_conv = MaskedConv1D(
                    out_channel, out_channel, 3,
                    padding=1, bias=(not with_ln), groups=out_channel
                )
                # layer norm for order (B C T)
                if with_ln:
                    if norm_first:
                        input_norm = LayerNorm(in_channels[i])
                    else:
                        input_norm = nn.Identity()

                    lateral_norm = LayerNorm(out_channel)
                    fpn_norm = LayerNorm(out_channel)
                else:
                    lateral_norm = nn.Identity()
                    fpn_norm = nn.Identity()

            self.input_norms.append(input_norm)
            self.lateral_convs.append(l_conv)
            self.lateral_norms.append(lateral_norm)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

        self.mask_features = MaskedConv1D(out_channel, out_channel, 3, padding=1, groups=out_channel)


    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        for idx in range(len(self.lateral_convs) - 1, -1, -1):
            x = inputs[idx]
            mask = fpn_masks[idx]

            input_norm = self.input_norms[idx]
            lateral_conv = self.lateral_convs[idx]
            lateral_norm = self.lateral_norms[idx]
            fpn_conv = self.fpn_convs[idx]
            fpn_norm = self.fpn_norms[idx]

            x = input_norm(x)
            if lateral_conv is None:
                y, _ = fpn_conv(x, mask)
                y = fpn_norm(y)
            else:
                cur_fpn, _ = lateral_conv(x, mask)
                cur_fpn = lateral_norm(cur_fpn)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, scale_factor=self.scale_factor, mode="nearest")
                y, _ = fpn_conv(y, mask)
                y = fpn_norm(y)
            
        out, out_mask = self.mask_features(y, fpn_masks[0])
        return out, out_mask

