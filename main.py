import diff_rl
import os
import time
from stable_baselines3.common.env_util import make_vec_env

def main(
		env_id,
		algo,
		n_envs,
		iter_num,
		seed,
		early_stop):

	algo_name = algo
	log_name = algo_name
	algo = eval('diff_rl.'+ algo)
	env_kwargs = None

	env = make_vec_env(env_id=env_id, n_envs=n_envs, env_kwargs=env_kwargs)
	# make experiment directory
	logdir = f"{env_id}/{log_name}/logs/{int(time.time())}/"
	modeldir = f"{env_id}/{log_name}/models/{int(time.time())}/"

	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	if early_stop:
		target_kl = 0.02
	else:
		target_kl = None
	model = algo(
	      		env, 
				verbose=1, 
				tensorboard_log=logdir, 
				seed=seed,
				target_kl=target_kl)

	for i in range(iter_num):
		model.learn(reset_num_timesteps=False, tb_log_name=f"{algo_name}")
		model.save(modeldir, f'{i * n_envs * model.n_steps}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Ant-v4') # 'Turtlebot-v2''Safexp-PointGoal1-v0'
    parser.add_argument('--algo', type=str, default='Diffusion_RL') 
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--iter_num', type=int, default=700) # Total_timestep = iter_num * n_envs * n_steps, here is 2000 * 4 * 20480 = 1.2e7
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true') # if no action, or said default if False, otherwise it's True
    args = parser.parse_args()

    main(
	    args.env_id, 
		args.algo, 
		args.n_envs, 
		args.iter_num, 
		args.seed,
		args.early_stop)