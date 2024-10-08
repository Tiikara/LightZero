from easydict import EasyDict
import time
import retro

env_id = 'Airstriker-Genesis'  # You can specify any Retro game here

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
update_per_collect = None
replay_ratio = 0.25
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
max_env_step = int(1e6)
reanalyze_ratio = 0.
batch_size = 64
# ==============================================================
# Размер контекста при обучении. Определяет размер максимальный размер контекста сети
# ==============================================================
num_unroll_steps = 10
# ==============================================================
# Определяет, сколько модель использует прошлых шагов.
# ==============================================================
infer_context_length = 4

observation_shape=(3, 64, 64)

# ====== only for debug =====
# collector_env_num = 2
# n_episode = 2
# evaluator_env_num = 2
# num_simulations = 5
# max_env_step = int(5e5)
# reanalyze_ratio = 0.
# batch_size = 2
# num_unroll_steps = 10
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

action_space_size = retro.make(game=env_id).action_space.shape[0]

retro_unizero_config = dict(
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=observation_shape,
        gray_scale=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        save_replay=True,
        replay_path='/mnt/d/source/LightZero/data_unizero/replay',
        continous_reward_wrapper=dict(
            enabled=False, # Enable only if the game requires a mandatory action. An agent in a game can't just do nothing
            reward=0.001,
            max_reward=0.25
        ),
        # TODO: only for debug
        # collect_max_episode_steps=int(50),
        # eval_max_episode_steps=int(50),
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            use_optimized_representation=True, # Use optimized version of RepresentationModel
            world_model_cfg=dict(
                max_blocks=num_unroll_steps,
                max_tokens=2 * num_unroll_steps,  # NOTE: each timestep has 2 tokens: obs and action
                context_length=2 * infer_context_length,
                device='cuda',
                # device='cpu',
                action_space_size=action_space_size,
                num_layers=4,
                num_heads=8,
                embed_dim=768,
                obs_type='image',
                env_num=max(collector_env_num, evaluator_env_num),
                caps_direction_loss_weight=2.,
                value_loss_weight=0.25,  # 0.25 - UniZero
                obs_loss_weight=10.  # 10. - UniZero
            ),
        ),
        learn=dict(
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=10_000,
                )
            ),
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        use_late_dropout=False,
        num_unroll_steps=num_unroll_steps,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='AdamW',
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
retro_unizero_config = EasyDict(retro_unizero_config)
main_config = retro_unizero_config

retro_unizero_create_config = dict(
    # env=dict(
    #     type='atari_lightzero_selfplay',
    #     import_names=['zoo.atari.envs.atari_self_play_env'],
    # ),
    env=dict(
        type='retro_lightzero',
        import_names=['zoo.retro.envs.retro_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='unizero',
        import_names=['lzero.policy.unizero'],
    ),
)
retro_unizero_create_config = EasyDict(retro_unizero_create_config)
create_config = retro_unizero_create_config

if __name__ == "__main__":
    # Define a list of seeds for multiple runs
    seeds = [int(time.time())]  # You can add more seed values here
    for seed in seeds:
        # Update exp_name to include the current seed
        main_config.exp_name = f'data_unizero/{env_id[:-14]}_stack1_unizero_upc{update_per_collect}-rr{replay_ratio}_H{num_unroll_steps}_bs{batch_size}_seed{seed}'
        from lzero.entry import train_unizero
        train_unizero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)
