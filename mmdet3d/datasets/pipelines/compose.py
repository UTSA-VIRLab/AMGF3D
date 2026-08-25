# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially. The pipeline registry of
    mmdet3d separates with mmdet, however, sometimes we may need to use mmdet's
    pipeline. So the class is rewritten to be able to use pipelines from both
    mmdet3d and mmdet.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        # Import transforms_3d here to avoid circular import
        # (transforms_3d imports Compose, so we can't import it at module level)
        # This ensures TR3DPointSample and other transforms are registered
        from . import transforms_3d  # noqa: F401
        
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform_type = transform['type']
                # Check if split_scope_key method exists
                if hasattr(PIPELINES, 'split_scope_key'):
                    try:
                        _, key = PIPELINES.split_scope_key(transform_type)
                    except (AttributeError, TypeError):
                        key = transform_type
                else:
                    key = transform_type
                
                # Try PIPELINES first (mmdet3d registry)
                if key in PIPELINES._module_dict.keys():
                    transform = build_from_cfg(transform, PIPELINES)
                else:
                    # If not in PIPELINES, check if it might be registered after late import
                    # Re-import transforms_3d to ensure all transforms are registered
                    from . import transforms_3d  # noqa: F401
                    
                    # Retry PIPELINES after import
                    if key in PIPELINES._module_dict.keys():
                        transform = build_from_cfg(transform, PIPELINES)
                    else:
                        # Try MMDET_PIPELINES as final fallback
                        try:
                            transform = build_from_cfg(transform, MMDET_PIPELINES)
                        except KeyError:
                            # Provide helpful error message
                            available = sorted(list(PIPELINES._module_dict.keys()))[:20]
                            raise KeyError(
                                f'{transform_type} is not in the pipeline registry. '
                                f'Available transforms in PIPELINES: {available}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
