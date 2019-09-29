import time, os, csv
import train_gymfc as tg
import eval_gymfc as eg
import exp_runner as er
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import tf_util as U
from sklearn.metrics import r2_score


def readExp(exp_path):
	# Load experiments
	exps = pd.read_csv(exp_path)
	exps = exps.drop(columns=['Notes','eps trained'])

	return exps
def cof(exps,ind):
	return [exps['cr'][ind],exps['ce'][ind],exps['cs'][ind],exps['cg'][ind]]

def loadExpEvals(exps):
	mes = []
	for i in range(len(exps['Model'])):
		me = eg.loadEval(exps['Model'][i])
		mes.append(me)

	return mes



def runEvals(exps):
#FINISH
# Run and save an evaluation for all trained models
	for i in range(len(exps['Model'])):
		me = eg.ModelEval(exps['Model'][i],'AttFC_GyroErr1_M4_Ep-v0')
		me.evalModel(50)
		me.saveEval()


def initRewards(exps):
# FINISH
# Creates rewards from untrained models to compare to trained models to assess reward elements learned
	me = eg.ModelEval(exps['Model'][i],'AttFC_GyroErr1_M4_Ep-v0')
	me.evalModel()
	me.saveEval(save_name='init'+str(i))

def rewardEval(exps, ctr):
#FINISH
# Compares trained reward values with untrained reward values of the same exp
	mes = loadExpEvals(exps) #Load evaluations
	avg_rwds = []
	ctr_avg_rwds = []
	avg_diff = []

#	for i in range(len(exps['Model'])):
#		for k in range(len(ctr.eps)):
#			_,_,_,_,_,ctrtots = computeReward(ctr.eps[k],cof(exps,i))
#			ctr_avg_rwds.append(ctrtots)
#		print(np.mean(ctr_avg_rwds, axis=0))

	for i in range(len(exps['Model'])): 
		rwds = []
		ctr_rwds = []
		rwd_dif = []
		for j in range(len(mes[i].eps)):
			# Calc untrained rewards as many times as trained reward eps
			_,_,_,_,_,ctrtots = computeReward(ctr.eps[j%len(mes[i].eps)],cof(exps,i))
			ctr_rwds.append(np.array(ctrtots))

			# Calc trained rewards
			_,_,_,_,_,tots = computeReward(mes[i].eps[j],cof(exps,i))
			rwds.append(np.array(tots))
		avg_rwds.append(np.mean(rwds,axis=0))
		ctr_avg_rwds.append(np.mean(ctr_rwds,axis=0))
	
	print(avg_rwds)
	print(ctr_avg_rwds)
	dif_avg = np.array(avg_rwds)-np.array(ctr_avg_rwds)
	perc_avg = np.abs((np.array(avg_rwds)-np.array(ctr_avg_rwds))/np.array(avg_rwds))
	print('dif: ', dif_avg)
	print('pec dif: ', perc_avg)


def riseTimes(mes):
# **SPEED** Average all maximum risetimes for all experiments
	rt = []
	for i in range(len(mes)):
		avg_rt = np.max([np.mean(mes[i].proc_eval.eps_r_rise),
			np.mean(mes[i].proc_eval.eps_p_rise),
			np.mean(mes[i].proc_eval.eps_y_rise)])
		rt.append(avg_rt)
	return rt

def errs(mes):
# **ERROR** Average all maximum risetimes for all experiments
	all_err = []
	for i in range(len(mes)):
		avg_err = np.sum([np.mean(mes[i].proc_eval.eps_avg_rerr),
			np.mean(mes[i].proc_eval.eps_avg_perr),
			np.mean(mes[i].proc_eval.eps_avg_yerr)])
		all_err.append(avg_err)
	return all_err

def eng(mes):
	all_eng = []
	for j in range(len(mes)):
		avg_acts = []
		for i in range(len(mes[j].eps)):
			avg_acts.append(np.sum(sum(np.abs(mes[j].eps[i]['actions']))))
		all_eng.append(np.mean(avg_acts))
	return all_eng


def evalExps(exp_path):
#FINISH
# Make full evaluation of experiments from sheet
	exps = readExp(exp_path) #Read exps from csv
	mes = loadExpEvals(exps) #Load evaluations

	


def computeReward(eps, coefs):
	""" Compute the reward """
	max_v = 6.2832
	thr = .0628 # .0628-99.5% | .1257-99% | .6283-95%
	speedFlag = 1
	act_min = np.array([-1,-1,-1,-1])
	act_max = np.array([1,1,1,1])
	rr,re,rs,rg,ra=[],[],[],[],[]
	cr,ce,cs,cg=coefs[0],coefs[1],coefs[2],coefs[3]
	for i in range(len(eps['actions'])):
		err = np.array([eps['roll_err'][i],eps['pitch_err'][i],eps['yaw_err'][i]])
#		r_r = -np.clip(np.sum(np.abs(err))/(max_v*3), 0, 1) # err
		r_r = -np.sum(np.square(err))
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
	coeff_filename = 'model_rcoeff_suc'
	exp_path = os.environ['GYMFC_EXP_MODELSDIR']+coeff_filename+'.csv'
	exps = readExp(exp_path)
	mes = loadExpEvals(exps)

	# Reward Learning Evaluation
#	ctr = eg.loadEval('ctr')
#	rewardEval(exps,ctr)

	# External Metrics
	rt = riseTimes(mes)
	exps['rise times'] = rt # add risetime to df
	mod_errs = errs(mes)
	exps['errors'] = mod_errs # add errors to df
	mod_eng = eng(mes)
	exps['energy'] = mod_eng
	print(exps)
	
	# r^2 scores of coefficients
#	r2s = [r2_score(exps['cr'],exps['errors']),
#		r2_score(exps['ce'],exps['energy']),
#		r2_score(exps['cs'],exps['rise times'])]
#	print(r2s)

	#Correlation matrix
	c_exps = exps.drop(columns='Model')
	c_exps = c_exps.astype('float')
	corr = c_exps.corr()
	print(corr)
	print('cr x errors:', corr['cr']['errors'])
	print('ce x energy:', corr['ce']['energy'])
	print('cs x rise times:', corr['cs']['rise times'])




