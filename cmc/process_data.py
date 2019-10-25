import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def plot_bar(data,heads,stat,clr):
	fig, ax = plt.subplots()
	N = data.shape
	ind = np.arange(1,N[1]+1)
	means = np.array(data.loc[stat])


	plt.bar(ind,means,color=clr)
	ax.set_xticks(ind)
	ax.set_xticklabels(heads)
	ax.set_title(stat)
	plt.xticks(rotation=45)
	plt.show()
	


def read_data(csv):
	data = pd.read_csv(csv, header=None, names = ['train_steps','nodes','layers','actor_batch','avg_r','avg_s','avg_e','avg_rr','std_r','std_s','std_e','std_rr','train_time'])
	data ['success']=data['avg_s']<990
	data ['e_s']=data['avg_e']/data['avg_s'] #average energy per average episode count
	return data


def data_stats(df,success=False):
	success_rate = sum(df['success'].astype(int))/len(df['success'])
	
	if success:
		df=df[df['success']]

	mean_s = df['avg_s'].mean()
	std_s = df['avg_s'].std()
	min_s = df['avg_s'].min()
	mean_stds = df['std_s'].mean()
	std_stds = df['std_s'].std()

	mean_e = df['avg_e'].mean()
	std_e = df['avg_e'].std()
	min_e = df['avg_e'].min()
	mean_stde = df['std_e'].mean()
	std_stde = df['std_e'].std()

	mean_r = df['avg_rr'].mean()
	std_r = df['avg_rr'].std()
	min_r = df['avg_rr'].min()
	mean_stdr = df['std_rr'].mean()
	std_stdr = df['std_rr'].std()

	mean_train_t = df['train_time'].mean()
	std_train_t = df['train_time'].std()
	min_train_t = df['train_time'].min()
	mean_e_s = df['e_s'].mean()
	ds = (mean_s,std_s,min_s,mean_stds,std_stds,
		mean_e,std_e,min_e,mean_stde,std_stde,
                mean_r,std_r,mean_stdr,std_stdr,
		mean_train_t,std_train_t,min_train_t,
		success_rate,
		mean_e_s)
	return ds


if __name__ == '__main__':
	reward_exp = {}
	exp_stats_ind = ['mean_s','std_s','min_s','mean_stds','std_stds',
		'mean_e','std_e','min_e','mean_stde','std_stde',
                'mean_r','std_r','min_r','mean_stdr','std_stdr',
		'mean_train_t','std_train_t','min_train_t',
		'success_rate',
		'mean_e_s']
	exp_stats = pd.DataFrame(index=exp_stats_ind)

	re_rr_rs_rg = '' #Set to name from cmc_train ex: '/home/user/CMC/rewardFunc.csv'

	csvs = (re_rr_rs_rg)
	exp_names = ['re_rr_rs_rg']
	for curr, exps in enumerate(exp_names):
		reward_exp[exps] = read_data(csvs[curr])
		exp_stats[exps] = data_stats(reward_exp[exps],success=True)
	print(exp_stats)

	plot_bar(exp_stats,exp_names,exp_stats_ind[0],'r')
	plot_bar(exp_stats,exp_names,exp_stats_ind[5],'r')
	plot_bar(exp_stats,exp_names,exp_stats_ind[10],'r')
	plot_bar(exp_stats,exp_names,exp_stats_ind[19],'r')
	plot_bar(exp_stats,exp_names,exp_stats_ind[15],'r')
	plot_bar(exp_stats,exp_names,exp_stats_ind[18],'g')


