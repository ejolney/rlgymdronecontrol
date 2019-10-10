import time, os, csv, math, sys

sys.path.insert(1, '/home/acit/gymfc/examples/controllers')
import run_iris_pid as rip

import train_gymfc as tg
import eval_gymfc as eg
import exp_runner as er
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from baselines.common import tf_util as U
from sklearn.metrics import r2_score
from statsmodels.multivariate.manova import MANOVA


def readExp(exp_path):
	# Load experiments
	exps = pd.read_csv(exp_path)
	exps = exps.drop(columns=['Notes','eps trained'])

	return exps
def cof(exps,ind):
	return [exps['cr'][ind],exps['ce'][ind],exps['cs'][ind],exps['cg'][ind]]

def loadExpEvals(exps):
	print('Loading Saved Evaluations...')
	mes = []
	for i in range(len(exps['Model'])):
#		print(exps['Model'][i], type(exps['Model'][i]))
		me = eg.loadEval(exps['Model'][i])
		mes.append(me)

	return mes

def viewResponses(mes,me_id=0):
# View Experiment responses
#	for i in range(len(mes[me_id].eps)):
	for i in range(2):
		print(mes[me_id].model_name, 'Episode: {}'.format(i))
		rpyPlot(mes[me_id], title='Step Response of RL Controller', ep=i)


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
# **ERROR** Sum of average errors
	all_err = []
	for i in range(len(mes)):
		avg_err = np.mean([np.array(mes[i].proc_eval.eps_avg_rerr)+
			np.array(mes[i].proc_eval.eps_avg_perr)+
			np.array(mes[i].proc_eval.eps_avg_yerr)])
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
	# Computes reward elements
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


def rpyPlot(me, ep=0, title=None):
	step_size=0.001
	threshold_percent=0.1
	end=1.

	# Get desired values
	rpy=[me.eps[ep]['droll_v'], me.eps[ep]['dpitch_v'], me.eps[ep]['dyaw_v']]
	desired=list(map(np.array,zip(*rpy)))
	desired=np.array(desired)
	print(desired[0])

	# Get actual values
	rpy=[me.eps[ep]['aroll_v'], me.eps[ep]['apitch_v'], me.eps[ep]['ayaw_v']]
	actual=list(map(np.array,zip(*rpy)))
	actual=np.array(actual)

	#actual = actual[:,:end,:]
	end_time = len(desired) * step_size
	t = np.arange(0, end_time, step_size)

	#actual = actual[:,:end,:]
	end_time = len(desired) * step_size
	t = np.arange(0, end_time, step_size)

	#desired = desired[:end]
	threshold = threshold_percent * desired

	plot_min = -math.radians(350)
	plot_max = math.radians(350)

	subplot_index = 3
	num_subplots = 3

	f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
	f.set_size_inches(10, 5)
	if title:
		plt.suptitle(title)
	ax[0].set_xlim([0, end_time])
	res_linewidth = 2
	linestyles = ["c", "m", "b", "g"]
	reflinestyle = "k--"
	error_linestyle = "r--"

	# Always
	ax[0].set_ylabel("Roll (rad/s)")
	ax[1].set_ylabel("Pitch (rad/s)")
	ax[2].set_ylabel("Yaw (rad/s)")

	ax[-1].set_xlabel("Time (s)")


	""" ROLL """
	# Highlight the starting x axis
	ax[0].axhline(0, color="#AAAAAA")
	ax[0].plot(t, desired[:,0], reflinestyle)
	ax[0].plot(t, desired[:,0] -  threshold[:,0] , error_linestyle, alpha=0.5)
	ax[0].plot(t, desired[:,0] +  threshold[:,0] , error_linestyle, alpha=0.5)

	r = actual[:,0]
	ax[0].plot(t[:len(r)], r, linewidth=res_linewidth, color="#ff0000")

	ax[0].grid(True)



	""" PITCH """

	ax[1].axhline(0, color="#AAAAAA")
	ax[1].plot(t, desired[:,1], reflinestyle)
	ax[1].plot(t, desired[:,1] -  threshold[:,1] , error_linestyle, alpha=0.5)
	ax[1].plot(t, desired[:,1] +  threshold[:,1] , error_linestyle, alpha=0.5)
	p = actual[:,1]
	ax[1].plot(t[:len(p)],p, linewidth=res_linewidth, color="#00ff00")
	ax[1].grid(True)


	""" YAW """
	ax[2].axhline(0, color="#AAAAAA")
	ax[2].plot(t, desired[:,2], reflinestyle)
	ax[2].plot(t, desired[:,2] -  threshold[:,2] , error_linestyle, alpha=0.5)
	ax[2].plot(t, desired[:,2] +  threshold[:,2] , error_linestyle, alpha=0.5)
	y = actual[:,2]
	ax[2].plot(t[:len(y)],y , linewidth=res_linewidth, color="#0000ff")
	ax[2].grid(True)

	plt.show()


def expandEvals(mes):
	aexps = {'model':[],'error':[],'rise_time':[],'energy':[]}
	aexps = pd.DataFrame(aexps)
	for k in range(len(mes)):
		#errors
		errs = list(np.array(mes[k].proc_eval.eps_avg_rerr)
			+np.array(mes[k].proc_eval.eps_avg_perr)
			+np.array(mes[k].proc_eval.eps_avg_yerr))
	#	print(errs)
		#energies
		engs = []
		for j in range(len(mes[k].eps)):
			engs.append(np.sum(sum(np.abs(mes[k].eps[j]['actions']))))
	#	print(engs)
		#rise times
		rises = np.max([np.array(mes[k].proc_eval.eps_r_rise),
			np.array(mes[k].proc_eval.eps_p_rise),
			np.array(mes[k].proc_eval.eps_y_rise)],axis=0)
	#	print(rises)
		#modelNames
		names = [mes[k].model_name]*len(errs)
	#	print(names)
		
		data={'model':names,'error':errs,'rise_time':rises,'energy':engs}
		data=pd.DataFrame(data)
	#	print(data)
		aexps=aexps.append(data)
	return aexps

def rlrlANOVA(mes):
	aexps = expandEvals(mes)

	print('-----RL Controller Error ANOVA-----')
	amod = ols('error ~ model', data=aexps).fit()
	atable = sm.stats.anova_lm(amod, typ=2)
	print(atable)
	print('-----RL Controller Speed ANOVA-----')
	amod = ols('rise_time ~ model', data=aexps).fit()
	atable = sm.stats.anova_lm(amod, typ=2)
	print(atable)
	print('-----RL Controller Energy ANOVA-----')
	amod = ols('energy ~ model', data=aexps).fit()
	atable = sm.stats.anova_lm(amod, typ=2)
	print(atable)

def rlrlRMANOVA(mes):
	# RL-RL ANOVA RM
	aexps = expandEvals(mes)

	print('********** RL Controller Error RMANOVA **********')
	aexps['s_id']=(np.array(aexps.index.values.tolist())+1).tolist()
	avrm = AnovaRM(aexps,'error','s_id',within=['model'])
	rma = avrm.fit()
	print(rma)

	print('********** RL Controller Error RMANOVA **********')
	aexps['s_id']=(np.array(aexps.index.values.tolist())+1).tolist()
	avrm = AnovaRM(aexps,'rise_time','s_id',within=['model'])
	rma = avrm.fit()
	print(rma)

	print('********** RL Controller Error RMANOVA **********')
	aexps['s_id']=(np.array(aexps.index.values.tolist())+1).tolist()
	avrm = AnovaRM(aexps,'energy','s_id',within=['model'])
	rma = avrm.fit()
	print(rma)

def rlpidttest(mes, pidmes):
	apidexp = expandEvals(pidmes)
	rlexp = expandEvals([mes[1]])
	rl_pid=rlexp.append(apidexp)

	print('********** PID RL, Paired T Test **********')
	print('----- Error -----')
	ttest, pval = stats.ttest_rel(apidexp['error'], rlexp['error'])
	print('t: ',ttest, ' p: ', pval)
	print('----- Speed -----')
	ttest, pval = stats.ttest_rel(apidexp['rise_time'], rlexp['rise_time'])
	print('t: ',ttest, ' p: ', pval)
	print('----- Energy -----')
	ttest, pval = stats.ttest_rel(apidexp['energy'], rlexp['energy'])
	print('t: ',ttest, ' p: ', pval)


def r2sExp(exps):
	# r^2 scores of coefficients
	r2s = [r2_score(exps['cr'],exps['errors']),
		r2_score(exps['ce'],exps['energy']),
		r2_score(exps['cs'],exps['rise times'])]
	print(r2s)

def corrsExp(exps):
	c_exps = exps.drop(columns='Model')
	c_exps = c_exps.astype('float')
	corr = c_exps.corr()
	print(corr)
	print('cr x errors:', corr['cr']['errors'])
	print('ce x energy:', corr['ce']['energy'])
	print('cs x rise_times:', corr['cs']['rise_times'])

def mvsExp(exps):
	#MANOVA
	mnv = MANOVA.from_formula('rise_times + errors + energy ~ ce', data=exps)
	print(mnv.mv_test())

	#Multiple Linear Regression
	est = ols(formula='rise_times ~ cr + ce + cs + cg', data=exps).fit()
	print(est.summary())
	est = ols(formula='errors ~ cr + ce + cs + cg', data=exps).fit()
	print(est.summary())
	est = ols(formula='energy ~ cr + ce + cs + cg', data=exps).fit()
	print(est.summary())

def extMetric(mes, exps):
	rt = riseTimes(mes)
	exps['rise_times'] = rt # add risetime to df
	mod_errs = errs(mes)
	exps['errors'] = mod_errs # add errors to df
	mod_eng = eng(mes)
	exps['energy'] = mod_eng # add energy to df

	return exps

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
		print('Evaluating reward for: {}'.format(exps['Model'][i])) 
		rwds = []
		ctr_rwds = []
		rwd_dif = []
		for j in range(len(mes[i].eps)):
			#print('eps {}'.format(j))
			# Calc untrained rewards as many times as trained reward eps
			_,_,_,_,_,ctrtots = computeReward(ctr.eps[j%len(mes[i].eps)],cof(exps,i))
			ctr_rwds.append(np.array(ctrtots))

			# Calc trained rewards
			_,_,_,_,_,tots = computeReward(mes[i].eps[j],cof(exps,i))
			rwds.append(np.array(tots))
		avg_rwds.append(np.mean(rwds,axis=0))
		ctr_avg_rwds.append(np.mean(ctr_rwds,axis=0))
	
#	print(avg_rwds)
#	print(ctr_avg_rwds)
	re_col = ['rr','re','rs','rg','tot']
	pre_col = ['prr','pre','prs','prg','ptot']
	dif_avg = np.array(avg_rwds)-np.array(ctr_avg_rwds)
	dif_avg = pd.DataFrame(dif_avg, columns=re_col)
	perc_avg = np.abs((np.array(avg_rwds)-np.array(ctr_avg_rwds))/np.array(avg_rwds))
	perc_avg = pd.DataFrame(perc_avg, columns=pre_col)
#	print('dif: ', dif_avg)
#	print('pec dif: ', perc_avg)

	#Save Reward Evaluation
	exps=exps.join(dif_avg)
	exps=exps.join(perc_avg)

	spath = os.environ['OPENAI_LOGDIR']+'reward_a.csv'
	print('saving to {}'.format(spath))
	exps.to_csv(spath)

def loadRewards(rpath):
	return pd.read_csv(rpath)

def printPDLT(df):
	pass

def main():
	tg.setVars()
	rip.loadEval('pid')
	# Coefficient file setup
	coeff_filename = 'model_rcoeff_exp_norm'
	exp_path = os.environ['OPENAI_LOGDIR']+coeff_filename+'.csv'
	# External Metric file setup
	ext_filename = 'ext_'+coeff_filename
	ext_path = os.environ['OPENAI_LOGDIR']+ext_filename+'.csv'
	exps = readExp(exp_path)

	# Create Evaluations for all models
#	runEvals(exps)

	# PID setup
	pidmes = []
	pidme = rip.loadEval('pid')
	pidmes.append(pidme)
	pidexp = {'Model':['pid'],'cr':[0],'ce':[0],'cs':[0],'cg':[0]}
	pidexp = pd.DataFrame(pidexp)
#	viewResponses(pidmes)

	ctrmes = []
	ctrme = rip.loadEval('ctr')
	ctrmes.append(ctrme)
	ctrexp = {'Model':['ctr'],'cr':[0],'ce':[0],'cs':[0],'cg':[0]}
	ctrexp = pd.DataFrame(ctrexp)

#	exps = pd.read_csv(ext_path, index_col=0) # Load exps instead of generating
#	print(exps)
	mes = loadExpEvals(exps)

	# External Metrics
	pidexp = extMetric(pidmes, pidexp)
	ctrexp = extMetric(ctrmes, ctrexp)
	exps = extMetric(mes, exps)

	# Save External Metrics
#	exps.to_csv(ext_path)

	tex = exps.append(pidexp)
	all_exp = tex.append(ctrexp)
	print(all_exp)



#	viewResponses(mes, 8)
#	for me_id in range(len(mes)):
#		viewResponses(mes, me_id)


	# Reward Learning Evaluation
#	ctr = eg.loadEval('ctr')
#	rewardEval(exps,ctr)
#	spath = os.environ['OPENAI_LOGDIR']+'reward_a.csv'
#	rexps = loadRewards(spath)



	#Correlation matrix
#	corrsExp(exps)

	#RL-RL ANOVA
#	rlrlANOVA(mes)

	# RL-RL ANOVA RM
#	rlrlRMANOVA(mes)

	#RL-PID Test
#	rlpidttest(mes, pidmes)

	#Multi-variate statistics
#	mvsExp(exps)
	return(all_exp)

if __name__ == '__main__':
#	main()
	tg.setVars()
	env_id = 'AttFC_GyroErr1_M4_Ep-v0'
#	rip.main(env_id, 17)
	me = eg.ModelEval('exp7',env_id)
	me.evalModel()
	me.proc_eval.plotResponse()

#	me = eg.ModelEval('exp11',env_id)
#	me.evalModel()
#	me.proc_eval.plotResponse()

#	me = eg.ModelEval('exp16',env_id)
#	me.evalModel()
#	me.proc_eval.plotResponse()

#	me = eg.ModelEval('exp14',env_id)
#	me.evalModel()
#	me.proc_eval.plotResponse()











