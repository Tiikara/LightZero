from .representation_network_unizero_capsnet_coord import RepresentationNetworkUniZeroCapsnetCoord

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
            observation_shape = observation_shape,
            num_res_blocks = num_res_blocks,
            activation = activation,
            norm_type = norm_type,
            embedding_dim = embedding_dim,
            group_size = group_size,
            num_capsules = model_config.num_capsules
        )
    else:
        raise 'Not supported'


