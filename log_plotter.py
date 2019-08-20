import os, csv
import pandas as pd
import matplotlib.pyplot as plt


def readLog(log_file):
	if os.path.isfile(log_file) and log_file.endswith('.csv'):
		log = pd.read_csv(log_file)
		return log
	else:
		print('No log file')
	
def plotLog(log):
	ep_avg_len = log['EpLenMean'].values  # average number of timesteps per ep since last iter

	# RL metric
	rew_avg = log['EpRewMean'].values  # average reward of episodes since last iter

	# Time keeping
	ep_this_iter= log['EpThisIter'].values  # number of episodes since last iter
	episodes = log['EpisodesSoFar'].values  # current number of episodes completed
	timesteps = log['TimestepsSoFar'].values  # current number of steps performed
	time_secs = log['TimeElapsed'].values  # seconds since learning started

	# Other Logged metrics
	ev_tdlam_before = log['ev_tdlam_before'].values
	ent_loss = log['loss_ent'].values
	kl_loss = log['loss_kl'].values
	entpen_pol_loss = log['loss_pol_entpen'].values
	surr_pol_loss = log['loss_pol_surr'].values
	vf_loss = log['loss_vf_loss'].values

	# Setup Plot
	fig, ax = plt.subplots()
	ax.plot(episodes, rew_avg)
	ax.set_title('Training Progress')
	ax.set_xlabel('Step')
	ax.set_ylabel('Avg_reward Per Episode')
	plt.show()

if __name__ == '__main__':
	log_filename = 'gymfc2019-8-14.csv'
	log_file = '/home/acit/gymlogs/training_stats/' + log_filename
	log = readLog(log_file)
	plotLog(log)
