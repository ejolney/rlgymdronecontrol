import time, os, csv
import train_gymfc as tg
import eval_gymfc as eg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import tf_util as U


def noescExp():
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0')
	cur_env = env[2]
	tp = tg.trainParams()
	pp = tg.policyParams()

	# Set up parameters
	tp.timesteps_per_actorbatch = 500
	tp.optim_batchsize = 32
	tp.optim_epochs = 5

	# Set training length
	num_eps = 3000
	tp.num_timesteps = num_eps*1000

	# Name Model
	tp.modelName('noesc01_rsg')

	#Run Training
	with tg.U.tf.Graph().as_default():
		tg.train(tp,pp,cur_env)

	# Model Evaluation
	me = eg.ModelEval(tp.model_name, cur_env)
	me.evalModel(20)
	me.saveEval()

def noescExpTest():
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0')
	cur_env = env[2]
	tp = tg.trainParams()
	pp = tg.policyParams()

	# Set up parameters
	tp.timesteps_per_actorbatch = 500
	tp.optim_batchsize = 32
	tp.optim_epochs = 5

	# Set training length
	num_eps = 1
	tp.num_timesteps = num_eps*1000

	# Name Model
	tp.modelName('noesc0t')

	#Run Training
	with tg.U.tf.Graph().as_default():
		tg.train(tp,pp,cur_env)

	# Model Evaluation
	me = eg.ModelEval(tp.model_name, cur_env)
	me.evalModel(2)
	me.saveEval()


def timeExp():
	env_id = 'AttFC_GyroErr1_M4_Ep-v0'
	exp_params = {
		'num_timesteps_exp' : [6000,8000,10000,12000],
		'timesteps_per_actorbatch_exp' : [100,250,500,1000,2000],
		'clip_param_exp' : [0.1, 0.2,0.5,0.9],
		'entcoeff_exp' : [0,0.1,0.5,0.9],
		'optim_epochs_exp' : [1,2,5,10,15],
		'optim_batchsize_exp' : [5, 10, 32, 50, 64, 100],
		'nodes_per_layer_exp' : [1,2,3],
		'num_layers_exp' : [10,20,32,64],
		'model_name': ['']
	}

	def_params = {
		'num_timesteps_exp' : 6000,
		'timesteps_per_actorbatch_exp' : 500,
		'clip_param_exp' : 0.1,
		'entcoeff_exp' : 0,
		'optim_epochs_exp' : 5,
		'optim_batchsize_exp' : 32,
		'nodes_per_layer_exp' : 2,
		'num_layers_exp' : 32,
		'model_name': ['']
	}

	num_exps = 0
	model_names = []

	# initialize experiment param recorder
	all_params = exp_params.copy()
	for akey in exp_params.keys(): all_params[akey] = []
	# Run Experiment
	for key in exp_params.keys(): 
		if key != 'model_name': num_exps = num_exps + len(exp_params[key])
		else: break
		for i in exp_params[key]:
			mn = 'time_x_'+key+'_'+str(i)  # Model name
			def_params['model_name'] = mn
			model_names.append(mn)
#			print(mn, key, i)

			# Setup experiment parameters
			cur_params = def_params.copy()
			cur_params[key] = i
			
			tp = tg.trainParams()
			tp.num_timesteps = cur_params['num_timesteps_exp']
			tp.timesteps_per_actorbatch = cur_params['timesteps_per_actorbatch_exp']
			tp.clip_param = cur_params['clip_param_exp']
			tp.entcoeff = cur_params['entcoeff_exp']
			tp.optim_epochs = cur_params['optim_epochs_exp']
			tp.optim_batchsize = cur_params['optim_batchsize_exp']
			tp.modelName(cur_params['model_name'])

			pp = tg.policyParams()
			pp.nodes_per_layer = cur_params['nodes_per_layer_exp']
			pp.num_layers = cur_params['num_layers_exp']

			for akey in all_params.keys(): all_params[akey].append(cur_params[akey])

			# Train and save models
			env = env_id
			with U.tf.Graph().as_default():
				tg.train(tp,pp,env_id)

#	# Line up parameters
#	for ind in range(num_exps):
#		param_list = [] 
#		for ky in all_params.keys(): 
#			param_list.append(all_params[ky][ind])
	return model_names

def analyzeTimeResults(mn_list, a_csv=None):
	time_data = []
	heads = ['model_name','num_timesteps','timesteps_per_actorbatch','clip_param','entcoeff',
		'optim_epochs','optim_batchsize','nodes_per_layer','num_layers']

	timex = {}
	for header in heads: 
		timex.update({ header : None })
	temptp = tg.trainParams()
	
	for mn in mn_list:


		mn_timex = timex.copy()
		# Get Training Data
		temptp.modelName(mn)
		temp_tl = eg.TrainLog(temptp.model_dir)

		# Parameter Data
		for header in heads: mn_timex[header] = getattr(temp_tl,header).to_numpy()


		# Training Time Data
		train_time = temp_tl.time_secs[-1:]
		mn_timex.update({'train_time' : train_time})
		time_data.append(mn_timex)


	# Save analysis
	if a_csv:
		t_keys = time_data[0].keys()

		try:
			with open(a_csv, 'w') as afile:
				writer = csv.DictWriter(afile, fieldnames = t_keys)
				writer.writeheader()
				for i in range(len(time_data)): writer.writerow(time_data[i])
		except IOError:
			print("I/O error with analysis csv")

def readTimex(timexpath):

	txdict = {}
	txd=[]
	with open(timexpath, 'r') as txcsv:
		reader = csv.reader(txcsv)
		heads = next(reader)
		txdict = txdict.fromkeys(heads,[])
		for _ in heads: txd.append([])
		for row in reader:
			for i in range(len(heads)): 
				txd[i].append(row[i])
		hdcnt = 0
		for hds in heads: 
			txdict[hds] = txd[hdcnt]
			hdcnt+=1
		return txdict


def timeAnalysis(timexpath, txdict=None):
	sns.set()
	z_buff = .00000000000001
	data = pd.read_csv(timexpath)
	data = data.applymap(lambda x: x.strip('[]')) # remove brackets from strings
	data = data.drop(columns='model_name')
	data = data.astype('float')
	
#	data = (data-data.mean())/data.std()  # Normalize data (mean normalization)
#	data = (data - data.min())/(data.max()-data.min()+z_buff)  #Normalize data (max-min norm)
	data = data.applymap(lambda x: x+z_buff) # turn 0s into non-zeros
	corr = data.corr()
	ax = sns.heatmap(corr)
	plt.show()
	
def readTx(txp):
	data = pd.read_csv(txp)
	data = data.applymap(lambda x: x.strip('[]')) # remove brackets from strings
	types = {}

	# Create pairs of values to training time in dict
	for i in range(1,len(data.keys())):
		typ = data.keys()[i]
		rows = data.loc[data['model_name'].str.find(typ)>0]
		types[typ] = rows[[typ,'train_time']].values.astype(float)
	return types

def checkR(model):
	me = eg.ModelEval(model,'AttFC_GyroErr1_M4_Ep-v0')
	me.evalModel(3)
	print('Average Reward: ', me.proc_eval.eps_avg_r)
	print('Max Rewards: ', me.proc_eval.eps_max_r)
	print('Rise times: ')
	print(me.proc_eval.eps_r_rise,me.proc_eval.eps_p_rise,me.proc_eval.eps_y_rise, sep='\n')
	print('-----Beginning-----')
	print('Rewards: ', 
		me.eps[0]['rewards'][0:4],
		me.eps[1]['rewards'][0:4],
		me.eps[2]['rewards'][0:4], sep='\n')
	print('-----End-----')
	print('Rewards: ', 
		me.eps[0]['rewards'][990:994],
		me.eps[1]['rewards'][990:994],
		me.eps[2]['rewards'][990:994], sep='\n')
	print('-------------')
	print('Desired: ', [me.eps[0]['droll_v'][0],me.eps[0]['dpitch_v'][0],me.eps[0]['dyaw_v'][0]])
	print('Total R: ', np.sum(me.eps[0]['rewards']),np.sum(me.eps[1]['rewards']),np.sum(me.eps[2]['rewards']))
#	me.proc_eval.plotResponse(0)
#	me.proc_eval.plotResponse(1)
#	me.proc_eval.plotResponse(2)

def compute_reward(eps):
	""" Compute the reward """
	max_v = 6.2832
	thr = .0628 # .0628-99.5% | .1257-99% | .6283-95%
	speedFlag = 1
	act_min = np.array([-1,-1,-1,-1])
	act_max = np.array([1,1,1,1])
	rr,re,rs,rg,ra=[],[],[],[],[]
	cr,ce,cs,cg=1,.01,.1,1
	for i in range(len(eps['actions'])):
		err = np.array([eps['roll_err'][i],eps['pitch_err'][i],eps['yaw_err'][i]])
		r_r = -np.clip(np.sum(np.abs(err))/(max_v*3), 0, 1) # err
		r_e = -1*np.sum(np.abs(np.clip(eps['actions'][i],act_min,act_max))) # energy
		r_s = -1 * speedFlag # speed
		if np.all(np.abs(err)<thr): 
			speedFlag = 0
			r_g = 1 - np.sum(np.abs(eps['actions'][i]-np.clip(eps['actions'][i],act_min,act_max))) # goal - saturation
			r_s = 0
		else: r_g = 0
		
		rr.append(r_r*cr)
		re.append(r_e*ce)
		rs.append(r_s*cs)
		rg.append(r_g*cg)
		ra.append((r_r*cr)+(r_e*ce)+(r_s*cs)+(r_g*cg)) # all rewards

	tots = [sum(rr),sum(re),sum(rs),sum(rg),sum(ra)]
	return rr,re,rs,rg,ra,tots


if __name__ == '__main__':
	tg.setVars()
#	checkR('noesc04_r')
#	noescExpTest()
	noescExp()
#	exp_id = 9
#	timeExp()
#	mn_list = [m for m in os.listdir(os.environ['GYMFC_EXP_MODELSDIR']) if m.startswith('time_x')]
#	acsv = '/home/acit/gymlogs/txp'+str(exp_id)+'.csv'
#	analyzeTimeResults(mn_list, a_csv=acsv)

