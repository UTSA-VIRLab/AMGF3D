try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from torch import nn

from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class TR3DNeck(BaseModule):
    """Original TR3D Neck with 2 FPN levels (skips level 0 and 3)."""

    def __init__(self, in_channels, out_channels):
        super(TR3DNeck, self).__init__()
        self._init_layers(in_channels[1:], out_channels)

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
                self.__setattr__(
                    f'out_block_{i}',
                    make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        x = x[1:]
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
                out = self.__getattr__(f'out_block_{i}')(x)
                outs.append(out)
        return outs[::-1]


def make_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels,
                                kernel_size=kernel_size, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_up_block(in_channels, out_channels, generative=False):
    conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
        else ME.MinkowskiConvolutionTranspose
    return nn.Sequential(
        conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


@NECKS.register_module()
class TR3DNeck4Level(BaseModule):
    """TR3D Neck with 4 FPN levels and pruning support.

    This neck processes all 4 backbone levels (unlike original TR3DNeck
    which skips level 0 and 3) and includes optional pruning for memory efficiency.

    This is a hybrid design combining:
    - FCAF3D's 4-level FPN structure with pruning
    - TR3D's simplified head design (used downstream)

    Args:
        in_channels (tuple): Input channels from backbone, e.g., (64, 128, 128, 128)
        out_channels (int): Output channels for each level, e.g., 128
        pts_threshold (int): Max points to keep per scene after pruning.
                            Set to -1 to disable pruning. Default: 100000
    """

    def __init__(self, in_channels, out_channels, pts_threshold=100000):
        super(TR3DNeck4Level, self).__init__()
        self.pts_threshold = pts_threshold
        self.n_levels = len(in_channels)
        self._init_layers(in_channels, out_channels)

    def _init_layers(self, in_channels, out_channels):
        """Initialize layers for 4-level FPN with pruning.

        Creates:
        - pruning: MinkowskiPruning layer
        - up_block_1, up_block_2, up_block_3: Upsampling blocks
        - lateral_block_0, lateral_block_1, lateral_block_2: Lateral connections
        - out_block_0, out_block_1, out_block_2, out_block_3: Output blocks for all levels
        """
        # Pruning layer (from FCAF3D)
        self.pruning = ME.MinkowskiPruning()

        n_levels = len(in_channels)  # Should be 4

        for i in range(n_levels):
            # Upsampling blocks (for levels 1, 2, 3 -> upsample to 0, 1, 2)
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))

            # Lateral blocks for feature fusion (for levels 0, 1, 2)
            # Level 3 (coarsest) doesn't need lateral block as it's the starting point
            if i < n_levels - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))

            # Output blocks for ALL levels (0, 1, 2, 3)
            self.__setattr__(
                f'out_block_{i}',
                make_block(in_channels[i], out_channels))

    def init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _prune(self, x, scores):
        """Prune sparse tensor based on classification scores.

        Keeps only top-k points per scene based on the maximum class score
        at each location. This is borrowed from FCAF3D to manage memory
        when processing fine-resolution levels.

        Args:
            x (SparseTensor): Input sparse tensor to prune
            scores (SparseTensor): Classification scores for pruning guidance

        Returns:
            SparseTensor: Pruned sparse tensor with at most pts_threshold points
        """
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)

            # Process each scene in the batch separately
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True

        x = self.pruning(x, prune_mask)
        return x

    def forward(self, x):
        """Forward pass through 4-level FPN.

        Processing order: coarsest (level 3) -> finest (level 0)
        - Start from level 3 (coarsest)
        - Upsample and add skip connections progressively
        - Apply pruning at finer levels to manage memory
        - Output predictions at all 4 levels

        Args:
            x (list[SparseTensor]): List of 4 SparseTensors from backbone
                [f0, f1, f2, f3] where f0 is finest and f3 is coarsest

        Returns:
            list[SparseTensor]: List of 4 output SparseTensors for head
                [out0, out1, out2, out3] maintaining the same order
        """
        # x = [f0, f1, f2, f3] - ALL 4 levels from backbone
        outs = []
        inputs = x
        n_levels = len(inputs)  # Should be 4
        prune_scores = None

        # Start from coarsest level (level 3)
        curr = inputs[-1]

        # Process from coarsest to finest: i = 3, 2, 1, 0
        for i in range(n_levels - 1, -1, -1):
            if i == n_levels - 1:
                # Level 3 (coarsest): just apply out_block, no upsampling needed
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)
                # Get initial prune scores from level 3 output
                prune_scores = self._get_prune_scores(out)
            else:
                # Levels 2, 1, 0: upsample, add skip, lateral, out
                curr = self.__getattr__(f'up_block_{i + 1}')(curr)
                curr = inputs[i] + curr

                # Apply pruning using scores from previous level
                if prune_scores is not None and self.pts_threshold > 0:
                    curr = self._prune(curr, prune_scores)

                # Lateral processing
                curr = self.__getattr__(f'lateral_block_{i}')(curr)

                # Output block
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)

                # Update prune scores for next (finer) level
                prune_scores = self._get_prune_scores(out)

        # Reverse to get [level0, level1, level2, level3] order
        return outs[::-1]

    def _get_prune_scores(self, out):
        """Create a SparseTensor with max features for pruning guidance.

        Args:
            out (SparseTensor): Output features from a level

        Returns:
            SparseTensor: SparseTensor with max feature value at each location
        """
        return ME.SparseTensor(
            out.F.max(dim=1, keepdim=True).values,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager)


@NECKS.register_module()
class TR3DNeck3Level(BaseModule):
    """TR3D Neck with 3 FPN levels (skips only level 0, keeps level 3).

    This is a compromise between original TR3D (2 levels) and 4-level:
    - Skips level 0 (finest) to save memory (like TR3D)
    - Keeps level 3 (coarsest) for large objects (like FCAF3D)
    - Outputs 3 levels: 1, 2, 3 (mapped to indices 0, 1, 2 in output)

    Args:
        in_channels (tuple): Input channels from backbone, e.g., (64, 128, 128, 128)
        out_channels (int): Output channels for each level, e.g., 128
    """

    def __init__(self, in_channels, out_channels):
        super(TR3DNeck3Level, self).__init__()
        # Skip first level (index 0), use levels 1, 2, 3
        self._init_layers(in_channels[1:], out_channels)

    def _init_layers(self, in_channels, out_channels):
        """Initialize layers for 3-level FPN.

        in_channels after slicing: (128, 128, 128) for levels 1, 2, 3

        Creates:
        - up_block_1, up_block_2: Upsampling blocks (level 3→2, level 2→1)
        - lateral_block_0, lateral_block_1: Lateral connections (for levels 1, 2)
        - out_block_0, out_block_1, out_block_2: Output blocks for all 3 levels
        """
        n_levels = len(in_channels)  # Should be 3

        for i in range(n_levels):
            # Upsampling blocks (for indices 1, 2 -> upsample to 0, 1)
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))

            # Lateral blocks for feature fusion (for indices 0, 1)
            # Index 2 (level 3, coarsest) doesn't need lateral block
            if i < n_levels - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))

            # Output blocks for ALL 3 levels
            self.__setattr__(
                f'out_block_{i}',
                make_block(in_channels[i], out_channels))

    def init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        """Forward pass through 3-level FPN.

        Args:
            x (list[SparseTensor]): List of 4 SparseTensors from backbone
                [f0, f1, f2, f3] where f0 is finest and f3 is coarsest

        Returns:
            list[SparseTensor]: List of 3 output SparseTensors for head
                [out1, out2, out3] (level 0 is skipped)
        """
        # Skip level 0, use levels 1, 2, 3
        x = x[1:]  # Now x = [f1, f2, f3]

        outs = []
        inputs = x
        n_levels = len(inputs)  # 3

        # Start from coarsest level (f3, index 2 in inputs)
        curr = inputs[-1]

        # Process from coarsest to finest: i = 2, 1, 0
        for i in range(n_levels - 1, -1, -1):
            if i == n_levels - 1:
                # Level 3 (coarsest, index 2): just apply out_block
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)
            else:
                # Levels 2, 1 (indices 1, 0): upsample, add skip, lateral, out
                curr = self.__getattr__(f'up_block_{i + 1}')(curr)
                curr = inputs[i] + curr
                curr = self.__getattr__(f'lateral_block_{i}')(curr)
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)

        # Reverse to get [level1, level2, level3] order (indices 0, 1, 2)
        return outs[::-1]


@NECKS.register_module()
class TR3DNeck3LevelFine(BaseModule):
    """TR3D Neck with 3 FPN levels (keeps level 0, skips level 3).

    This variant adds back the finest level (level 0) from FCAF3D:
    - Keeps level 0 (finest) for small objects
    - Keeps levels 1, 2 (like TR3D)
    - Skips level 3 (coarsest) to reduce computation
    - Outputs 3 levels: 0, 1, 2

    This is useful for detecting small objects that benefit from
    fine-grained features at level 0.

    Args:
        in_channels (tuple): Input channels from backbone, e.g., (64, 128, 128, 128)
        out_channels (int): Output channels for each level, e.g., 128
        pts_threshold (int): Max points to keep per scene after pruning.
                            Set to -1 to disable pruning. Default: 100000
    """

    def __init__(self, in_channels, out_channels, pts_threshold=100000):
        super(TR3DNeck3LevelFine, self).__init__()
        self.pts_threshold = pts_threshold
        # Use only first 3 levels (0, 1, 2), skip level 3
        self._init_layers(in_channels[:3], out_channels)

    def _init_layers(self, in_channels, out_channels):
        """Initialize layers for 3-level FPN (levels 0, 1, 2).

        in_channels: (64, 128, 128) for levels 0, 1, 2

        Creates:
        - pruning: MinkowskiPruning layer for memory management
        - up_block_1, up_block_2: Upsampling blocks (level 2→1, level 1→0)
        - lateral_block_0, lateral_block_1: Lateral connections (for levels 0, 1)
        - out_block_0, out_block_1, out_block_2: Output blocks for all 3 levels
        """
        self.pruning = ME.MinkowskiPruning()
        n_levels = len(in_channels)  # Should be 3

        for i in range(n_levels):
            # Upsampling blocks (for indices 1, 2 -> upsample to 0, 1)
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))

            # Lateral blocks for feature fusion (for indices 0, 1)
            # Index 2 (level 2, coarsest in this config) doesn't need lateral block
            if i < n_levels - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))

            # Output blocks for ALL 3 levels
            self.__setattr__(
                f'out_block_{i}',
                make_block(in_channels[i], out_channels))

    def init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _prune(self, x, scores):
        """Prune sparse tensor based on classification scores.

        Keeps only top-k points per scene based on the maximum class score
        at each location. Essential for level 0 which has many points.

        Args:
            x (SparseTensor): Input sparse tensor to prune
            scores (SparseTensor): Classification scores for pruning guidance

        Returns:
            SparseTensor: Pruned sparse tensor with at most pts_threshold points
        """
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)

            # Process each scene in the batch separately
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True

        x = self.pruning(x, prune_mask)
        return x

    def _get_prune_scores(self, out):
        """Create a SparseTensor with max features for pruning guidance."""
        return ME.SparseTensor(
            out.F.max(dim=1, keepdim=True).values,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager)

    def forward(self, x):
        """Forward pass through 3-level FPN (levels 0, 1, 2).

        Processing order: level 2 -> level 1 -> level 0
        - Start from level 2 (coarsest in this config)
        - Upsample and add skip connections progressively
        - Apply pruning at finer levels to manage memory
        - Output predictions at levels 0, 1, 2

        Args:
            x (list[SparseTensor]): List of 4 SparseTensors from backbone
                [f0, f1, f2, f3] where f0 is finest and f3 is coarsest

        Returns:
            list[SparseTensor]: List of 3 output SparseTensors for head
                [out0, out1, out2] (level 3 is skipped)
        """
        # Use only levels 0, 1, 2 (skip level 3)
        inputs = x[:3]  # Now inputs = [f0, f1, f2]
        n_levels = len(inputs)  # 3
        outs = []
        prune_scores = None

        # Start from coarsest level in this config (level 2, index 2)
        curr = inputs[-1]

        # Process from coarsest to finest: i = 2, 1, 0
        for i in range(n_levels - 1, -1, -1):
            if i == n_levels - 1:
                # Level 2 (coarsest in this config): just apply out_block
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)
                # Get initial prune scores from level 2 output
                prune_scores = self._get_prune_scores(out)
            else:
                # Levels 1, 0: upsample, add skip, prune, lateral, out
                curr = self.__getattr__(f'up_block_{i + 1}')(curr)
                curr = inputs[i] + curr

                # Apply pruning using scores from previous level
                # Critical for level 0 which has many points
                if prune_scores is not None and self.pts_threshold > 0:
                    curr = self._prune(curr, prune_scores)

                # Lateral processing
                curr = self.__getattr__(f'lateral_block_{i}')(curr)

                # Output block
                out = self.__getattr__(f'out_block_{i}')(curr)
                outs.append(out)

                # Update prune scores for next (finer) level
                prune_scores = self._get_prune_scores(out)

        # Reverse to get [level0, level1, level2] order
        return outs[::-1]
