from .representation_network_unizero_capsnet_coord import RepresentationNetworkUniZeroCapsnetCoord
from .representation_network_unizero_capsnet import RepresentationNetworkUniZeroCapsnet
from .representation_network_unizero_capsnet_forward import RepresentationNetworkUniZeroCapsnetForward
from .representation_network_unizero_capsem_only import RepresentationNetworkUniZeroCapsSEMOnly
from .representation_network_unizero_capsnet_pos import RepresentationNetworkUniZeroCapsnetPositional
from .representation_network_unizero_pos import RepresentationNetworkUniZeroPositional
from .representation_network_unizero_capsnet_squashless import RepresentationNetworkUniZeroCapsnetSquashless
from .representation_network_unizero_capsnet_res_downsample import \
    RepresentationNetworkUniZeroCapsnetResDownsample


def build_representation_network_unizero(
        observation_shape,
        num_res_blocks,
        activation,
        norm_type,
        embedding_dim,
        group_size,
        model_config
):
    if model_config.type == 'capsnet_coord':
        return RepresentationNetworkUniZeroCapsnetCoord(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'capsnet':
        return RepresentationNetworkUniZeroCapsnet(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'capsnet_forward':
        return RepresentationNetworkUniZeroCapsnetForward(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'capsem_only':
        return RepresentationNetworkUniZeroCapsSEMOnly(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'capsnet_pos':
        return RepresentationNetworkUniZeroCapsnetPositional(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'base_pos':
        return RepresentationNetworkUniZeroPositional(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size
        )
    elif model_config.type == 'capsnet_squashless':
        return RepresentationNetworkUniZeroCapsnetSquashless(
            observation_shape=observation_shape,
            num_res_blocks=num_res_blocks,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules
        )
    elif model_config.type == 'capsnet_res_downsample':
        return RepresentationNetworkUniZeroCapsnetResDownsample(
            observation_shape=observation_shape,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=embedding_dim,
            group_size=group_size,
            num_capsules=model_config.num_capsules,
            use_coords=model_config.use_coords,
            start_channels=model_config.start_channels
        )
    else:
        raise 'Not supported'
