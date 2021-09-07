#!/usr/bin/env python3
import numpy as np
import os
from glob import glob

_seed=20201205
decomposed_trainset = "data/CIFAR10_train.decomposed"
modelfile = "model/CIFAR10_svrandom+.pt" 
result_folder = "result/"


def get_labelgroup(idset, fileroot="data/"):
	tb = {}
	if idset == "CIFAR100":
		tb = np.load(fileroot + "CIFAR100_class_lookup.npy",allow_pickle=True).item()
	else:
		with open(fileroot + "{}_class_lookup.csv".format(idset), "r") as f:
			f.readline() #label,group
			for l in f:
				tmp = [int(x) for x in l.split(",")]
				tb[tmp[0]] = tmp[1]

	return tb

def get_macrolabel(label_lookup, labels): 
	labels_macro = []
	for i in range(len(labels)): labels_macro.append( label_lookup[labels[i]] )  
	return labels_macro

def seed():
	np.random.seed(_seed)

## evaluate routines
class Blindtest():

	def __init__(self):	
		# config
		self.testsets = {
				"CIFAR10": ["CIFAR10", "LSUN", "SVHN", "GTSDB", "FAKE-3", "UNIFORM-3"], 
				"CIFAR100": ["CIFAR100", "LSUN", "SVHN", "GTSDB", "FAKE-3", "UNIFORM-3"], 
				"GTSDB": ["GTSDB" ,"CIFAR10", "CIFAR100", "LSUN", "SVHN", "FAKE-3", "UNIFORM-3"]
				}
		self.networks = {"CIFAR10":"_resnet18", "CIFAR100":"_resnet101", "GTSDB":"_resnet18"}
		self.methods = {"fast":"Fast-FGSM", "halftone":"Halftone", "mixup":"Mixup", 
						"svd2":"SVrandom+", "none":"Standard"}

		self.attacks = {"none":None, "fgsm":(2), "pgd":(2,2), "cw2":(1), 
						"deepfool":(5,50), "noise":(10,1) }
		self.lookup_table = {
			"CIFAR100" : get_labelgroup("CIFAR100")["finec"],
			"CIFAR10" : get_labelgroup("CIFAR10"),
			"GTSDB" : get_labelgroup("GTSDB")
		}
		self.alpha = {"CIFAR100": 0.8, "CIFAR10":0.95, "GTSDB":0.95}		

		if os.path.isdir(result_folder) == False:
			input_test = input("Run new test? [Y|N] ")
			if input_test.strip()=="Y": self.runtest()


	def get_oodd_threshold(self, resultfile, fpr_level, alpha):	
		content = np.load(resultfile, allow_pickle=True).item()		
		truth_id, pred_id = content["objtrue"], content["objpred"]

		scores_correct = []
		if "oodscore_parent" not in content.keys():	
			for i in range(len(content["oodscore"])):
				if pred_id[i] == truth_id[i]: scores_correct.append(content["oodscore"][i])		
		else:
			truth_id_parent, pred_id_parent = content["objtrue_parent"], content["objpred_parent"]
			scores_id = [content["oodscore"][x]*alpha + (1.-alpha)*content["oodscore_parent"][x] \
														for x in range(len(content["oodscore"]))]
			scores_correct = []
			for i in range(len(scores_id)):
				if pred_id[i] == truth_id[i] and pred_id_parent[i] == truth_id_parent[i] :	
					scores_correct.append(scores_id[i])		

		return np.percentile(scores_correct, fpr_level)
		
	def get_setresult(self, attackid,  oodd_threshold, trainset, method):

		def _dual_match(trainset, pred, pred_parent):
			if pred_parent == self.lookup_table[trainset][pred]: return True
			else: return False

		oodd_tpr, cla_err = [], []

		# fetch test results
		for testset in self.testsets[trainset]:
			if method!="none":
				restult_file = result_folder +  "{}_{}{}_{}_{}.npy".format(\
												trainset, method, self.networks[trainset], testset, attackid)
			else:
				restult_file = result_folder +  "{}{}_{}_{}.npy".format(\
												trainset, self.networks[trainset], testset, attackid)				

			if os.path.isfile(restult_file) == False: 						
				print("Miss test result file {}".format(restult_file))

			else:
				content = np.load(restult_file, allow_pickle=True).item()			
				objpred = content["objpred"]
				objtruth = content["objtrue"]	
				if "oodscore_parent" not in content.keys():
					oodscore = content["oodscore"]	
				else: 
					objpred_parent = content["objpred_parent"]				
					oodscore = [content["oodscore"][i]*self.alpha[trainset] + \
										(1-self.alpha[trainset])*content["oodscore_parent"][i] \
										for i in range(len(content["oodscore"]))]

				ood_pred, ood_truth, cla_pred, cla_truth = [], [], [], []
				for j in range(len(oodscore)):
					# classification
					cla_pred.append(objpred[j])
					if testset!=trainset: 
						# oodset
						cla_truth.append(-1) # objtruth_parent always -1
						ood_truth.append(1)
					else: 
						ood_truth.append(0)
						cla_truth.append(objtruth[j])						

					# ood dtection
					# oodscore: idscore shall be greater than oodscore
					# ood_truth : 1/ood, 0/id
					if oodd_threshold is not None:
						if "objpred_parent" in locals():
							if oodscore[j] > oodd_threshold and \
								_dual_match(trainset, objpred[j], objpred_parent[j]):
									ood_pred.append(0)
							else: ood_pred.append(1)
						else:
							if oodscore[j] > oodd_threshold: ood_pred.append(0)
							else: ood_pred.append(1)

				cla_error_count = 0
				ood_correct_count = 0
				if oodd_threshold is not None:
					for x in range(len(ood_pred)):
						if ood_pred[x] == ood_truth[x] : ood_correct_count +=1
						if ood_pred[x] == 0 and cla_pred[x] != cla_truth[x]: cla_error_count += 1
				else:
					for x in range(len(cla_truth)):
						if cla_pred[x] != cla_truth[x]: cla_error_count += 1

				cla_err.append(cla_error_count/len(cla_truth))

				# skip ID and Adv. for TPR of ood dtection
				if oodd_threshold is not None and testset!=trainset:
					oodd_tpr.append(ood_correct_count/len(oodscore))


		if oodd_threshold is not None:
			return np.mean(oodd_tpr), np.mean(cla_err) 
		else:
			return None, np.mean(cla_err)

	def result_overall(self, oodd_fpr_level):
		result_by_classifier = {}
		for trainset in self.testsets.keys():
			result_by_method = {}
			for method in self.methods.keys():
				tpr, claerror = [], []
				if method != "none":
					result_file = result_folder + "{}_{}{}_{}_none.npy".format(\
												trainset, method, self.networks[trainset], trainset)
				else:
					result_file = result_folder + "{}{}_{}_none.npy".format(\
												trainset, self.networks[trainset], trainset)
				if oodd_fpr_level > 0:
					oodd_threshold = self.get_oodd_threshold(\
											result_file, oodd_fpr_level, self.alpha[trainset])
				else:
					oodd_threshold = None

				for attackid in self.attacks.keys():
					 _tpr, _claerror = self.get_setresult(attackid, oodd_threshold, trainset, method)
					 if _tpr is not None: tpr.append(_tpr)
					 claerror.append(_claerror)

				result_by_method[method] = (np.mean(tpr) if len(tpr) >0 else None, np.mean(claerror))

			result_by_classifier[trainset] = result_by_method

		return result_by_classifier

	def result_advdefense(self):

		def _acc(result_file):
			content = np.load(result_file, allow_pickle=True).item()
			pred_id = content["objpred"]
			truth_id = content["objtrue"]
			cla_correct = 0
			for i in range(len(pred_id)):
				if truth_id[i] == pred_id[i]: cla_correct+=1
			return  cla_correct/len(truth_id)
		
		result_by_classifier = {}
		for trainset in self.testsets.keys():

			result_by_attack = {}
			for attackid in self.attacks.keys():
				result_by_method = {}
				for method in self.methods.keys():
					if method=="none": continue

					result_file = result_folder + "{}_{}{}_{}_{}.npy".format(trainset, method, \
														self.networks[trainset], trainset, attackid)
					result_by_method[method]  = _acc(result_file)

				result_by_attack[attackid] = result_by_method
			result_by_classifier[trainset] = result_by_attack	

		return result_by_classifier


	def runtest(self):
		seed()
		#TODO
