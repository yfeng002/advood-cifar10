'''
Classes and routines for evaluate

'''
#!/usr/bin/env python3
import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from network import ResNet18Dual
import torchattacks

_seed=20201205
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
oodd_fpr_levels = [0,10,15,20]
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
	labels_parent = []
	for i in range(len(labels)): 
		labels_parent.append( label_lookup[labels[i]] )  
	return labels_parent

def seed():
	np.random.seed(_seed)
	torch.manual_seed(_seed)
	torch.cuda.manual_seed(_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

## Evaluate routines
class Blindtest():

	def __init__(self, args):	
		# Config
		self.testsets = {
				"CIFAR10": ["CIFAR10", "SVHN", "LSUN", "GTSDB",  "FAKE-3", "UNIFORM-3"], 
				"CIFAR100": ["CIFAR100", "LSUN", "SVHN", "GTSDB", "FAKE-3", "UNIFORM-3"], 
				"GTSDB": ["GTSDB" ,"CIFAR10", "CIFAR100", "LSUN", "SVHN", "FAKE-3", "UNIFORM-3"]
				}
		self.networks = {"CIFAR10":"_resnet18", "CIFAR100":"_resnet101", "GTSDB":"_resnet18"}
		self.methods = {"fast":"Fast-FGSM", "halftone":"Halftone", "mixup":"Mixup", 
						"svd2":"SVrandom+", "none":"Standard"}

		self.attacks = {"none":None, "fgsm":(2,), "pgd":(2,2), "cw2":(1,), 
						"deepfool":None, "noise":(10,) }
		self.attack_samples = {"none":None, "fgsm":None, "pgd":None, "cw2":500, "deepfool":500, "noise":None}

		self.lookup_table = {
			"CIFAR100" : get_labelgroup("CIFAR100")["finec"],
			"CIFAR10" : get_labelgroup("CIFAR10"),
			"GTSDB" : get_labelgroup("GTSDB")
		}
		self.alpha = {"CIFAR100": 0.8, "CIFAR10":0.95, "GTSDB":0.95}		

		if args.model is not None:
			print("\nDownload test sets and run new tests... (May take half hour to complete on GPU.)")
			self.newtest("model/"+args.model)

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
				if method == "none": continue
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

	def newtest(self, modelfile):
		seed()
		batchsize = 10
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")	
		trainset = "CIFAR10"

		# load model 
		model = ResNet18Dual(3, 10, 2, False)
		checkpoint = torch.load(modelfile, map_location=device)
		model.load_state_dict(checkpoint['net'])
		model.to(device)
		model.eval() 

		for testset in self.testsets[trainset]:
			testloader = get_testloader(testset, batchsize)
			# skip if test dataset is not ready
			if testloader is None: 
				print("{}'s test set is not found in data/ folder. Skip".format(testset))
				continue
			else:
				print("Working on test set {}".format(testset))

			for attackid in self.attacks.keys():
				print(attackid.upper())
				numofsamples2test = self.attack_samples[attackid]
				if attackid=="fgsm":
					atk = torchattacks.FGSM(model, eps=self.attacks[attackid][0]/255.)
				elif attackid=="pgd":
					atk = torchattacks.PGD(model, eps=self.attacks[attackid][0]/255.* self.attacks[attackid][1],
									steps=self.attacks[attackid][1])
				elif attackid=="cw2":
					atk = torchattacks.CW(model, c=self.attacks[attackid][0])
					self.attack_samples
				elif attackid=="deepfool":
					atk = torchattacks.DeepFool(model)		

				# Local variables to hold batch result
				true_class, pred_class = [], []
				true_parent_class, pred_parent_class = [], []
				oodscore, oodscore_parent = [], []	
				# Iterate through batches
				for _, (data, target) in enumerate(testloader):
					# Number of classes in an OOD testset could be greater than that in CIFAR10
					# Or an OOD testset may not have label 	
					# For all OOD testset, set target to class 0 
					if testset != trainset: target = torch.zeros(target.size()).long()

					data, target = data.to(device), target.to(device)

					if attackid in ["fgsm", "pgd", "cw2", "deepfool"]:
						data_adv = atk(data, target.to(device))	
					elif attackid=="noise":	
						data_adv = data + torch.FloatTensor(*data.shape).uniform_(\
											-self.attacks[attackid][0]/255., 
											self.attacks[attackid][0]/255.).to(device)
					else:
						data_adv = data

					data_adv = transforms.Normalize(cifar10_mean, cifar10_std)(data_adv)	
					outputs = model(data_adv).detach()	
					val, predicted = outputs.max(1)
					outputs_parent = model.forward_parent(data_adv).detach()
					val_parent, predicted_parent = outputs_parent.max(1)

					if testset == trainset:
						target_parent = get_macrolabel(self.lookup_table[trainset], target.cpu().numpy())

					for j in range(data_adv.size(0)):
						pred_class.append(predicted[j].item())
						if testset == trainset:
							true_class.append(target[j].item())
						else:
							true_class.append(-1)	

						pred_parent_class.append(predicted_parent[j].item())
						if testset == trainset:
							true_parent_class.append(target_parent[j])
						else:
							true_parent_class.append(-1)

						oodscore.append( val[j].item() )
						oodscore_parent.append( val_parent[j].item() )	

					# Early exit for very slow attack 
					if numofsamples2test is not None and len(true_class) > numofsamples2test: 
						oodscore = oodscore[:numofsamples2test]
						oodscore_parent = oodscore_parent[:numofsamples2test]
						true_class = true_class[:numofsamples2test]
						pred_class = pred_class[:numofsamples2test]
						true_parent_class = true_parent_class[:numofsamples2test]
						pred_parent_class = pred_parent_class[:numofsamples2test]
						break				
				#
				outputfile = result_folder +  "{}_svd2{}_{}_{}.npy".\
										format(trainset, self.networks[trainset], testset, attackid)
				np.save(outputfile, {
								"oodscore": oodscore, "oodscore_parent":oodscore_parent,
								"objtrue": true_class, "objpred": pred_class,
								"objtrue_parent":true_parent_class, "objpred_parent":pred_parent_class,
								 })

# New test
class GTSDB(Dataset):
	def __init__(self, path, transform):
		self.transform = transform

		self.imagefiles = []
		self.label = []
		#path_prefix = '/'.join(path.split("/")[:-1]) + '/'
		with open(path, 'r') as f:
			f.readline()
			for l in f:
				tmp = l.rstrip('\n').split(',')
				self.label.append(int(tmp[-2]))
				self.imagefiles.append(tmp[-1])

	def __len__(self):
		return len(self.imagefiles)

	def __getitem__(self, index):
		label = self.label[index]
		image = Image.open(self.imagefiles[index]) 
		image = self.transform(image)
		return image, label

class LSUN(Dataset):
	def __init__(self, path, transform):
		self.transform = transform	
		self.imagefiles = []
		with open(path, 'r') as f:
			for l in f:
				self.imagefiles.append(l.rstrip('\n'))

	def __len__(self):
		return len(self.imagefiles)

	def __getitem__(self, index):
		label = -1 # fake label not used
		image = Image.open(self.imagefiles[index]) 
		image = self.transform(image)
		return image, label

class UniformData(Dataset):
	def __init__(self, transform, numofsamples, img_channel, img_size):

		self.numofsamples = numofsamples
		self.images = torch.empty((numofsamples, img_channel, img_size, img_size))
		dynamic_range = 0.01 

		for i in range(numofsamples):
			for c in range(img_channel):
				mean = np.random.randint(0,250)
				data = np.ones((img_size, img_size)) * mean + \
						np.random.rand(img_size, img_size) * (mean*dynamic_range)
				self.images[i,c] = transform(data / 255.)

	def __len__(self):
		return self.numofsamples

	def __getitem__(self, index):
		return self.images[index], 0 # fake label not used

def get_testloader(dataset, batch_size):
	#
	db_needresize = ["LSUN", "GTSDB"]

	dataloader = None
	input_size = 32
	pads = 2
	# No data augmentation for testing
	if dataset in db_needresize:
		transform = transforms.Compose([
						transforms.Resize((input_size, input_size)),				
						transforms.ToTensor()])
	else:
		transform = transforms.Compose([transforms.ToTensor()])

	#
	if dataset=="CIFAR10":
		datasetobj = datasets.CIFAR10(root='data/CIFAR10/', train=False, download=True, transform=transform)
		dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)

	elif dataset=="SVHN":
		datasetobj = datasets.SVHN(root='data/SVHN/', split='test', download=True, transform=transform)
		dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)
	
	elif dataset=="FAKE-3": # Gaussian
		datasetobj = datasets.FakeData(10000, (3, input_size, input_size), num_classes=1, 
										transform=transforms.Compose([transforms.ToTensor(),]) )
		dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)		

	elif dataset=="UNIFORM-3": 
		datasetobj = UniformData(transform, 10000, 3, 32)
		dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)	

	elif dataset=="GTSDB":
		if os.path.isdir("data/GTSDB_Test/"):
			datasetobj = GTSDB('data/GTSDB_test.csv', transform=transform)
			dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)
		else:
			print("No GTSDB images found in data/GTSDB_Test/. Skip.")	
			dataloader = None

	elif dataset=="LSUN":
		if os.path.isdir("data/LSUN_Test/"):
			datasetobj = LSUN('data/LSUN_test.csv', transform=transform)
			dataloader = torch.utils.data.DataLoader(datasetobj, batch_size=batch_size, shuffle=True)
		else:
			print("No LSUN images found in data/LSUN_Test/. Skip.")	
			dataloader = None			

	else:
		assert False, "{} not supported".format(dataset)

	return dataloader



'''
## Train routines
from numpy.linalg import svd
decomposed_trainset = "data/CIFAR10_train.decomposed" 

class SVDataset(Dataset):
	def __init__(self, dataset, train):

		if train and os.path.isfile(decomposed_trainset+".npy") == False: 
			preprocess_trainset()

		content = np.load(decomposed_trainset+".npy", allow_pickle=True).item()
		self.U = content["U"]
		self.S = content["S"]
		self.V = content["V"]
		self.label = content["label"]

		label_lookup = get_labelgroup(dataset)
		if label_lookup is None:
			self.label_parent = []
		else:
			self.label_parent = get_macrolabel(label_lookup, self.label)		

	def __len__(self):
		return len(self.label)

	def __getitem__(self, index):
		if len(self.label_parent)==0:
			return self.U[index], self.S[index], self.V[index], self.label[index]
		else:
			return self.U[index], self.S[index], self.V[index], self.label[index], self.label_parent[index]

def augmentation(U, S, V, mean, std, augment_param):

	def _reconstruction(_u, _s, _v, mean, std):
		(_c,_h,_w) = _u.shape
		mat = np.zeros((_c,_h,_w))

		transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),				
			transforms.RandomCrop(_h, padding=4),				
			transforms.Normalize(mean, std),
			])	
			
		for ch in range(_c):
			for r in range(_h):		
				mat[ch, r, :] = np.multiply(_u[ch, r], _s[ch])
		mat = np.matmul(mat, _v)
		mat = transform(mat.float())
		return mat	
		
	(batchsize, C, H, W) = U.shape
	data = torch.empty(U.shape)

	# randomly pick a method
	augments = len(augment_param.keys())
	i = np.random.randint(augments) if augments > 1 else 0
	
	# reduce (drop) bottom U and V spaces
	if i == 1:
		bottomx = int(W * augment_param["reduce_bottomx"])
		for j in range(batchsize):
			m = np.random.randint(bottomx, W)
			data[j] = _reconstruction(U[j,:,:,:m], S[j,:,:m], V[j,:,:m,:], mean, std)
			
	# swap U and V subspaces of 2 images
	elif i == 0:
		bottomx = int(W * augment_param["pairswap_bottomx"])		
		if batchsize%2 !=0:
			U, S, V = U[:-1], S[:-1], V[:-1]
			batchsize-=1
		
		# split the batch 
		swap_index = np.arange(batchsize)	
		np.random.shuffle(swap_index)
		K = int(batchsize/2)		
		
		for j in range(K):
			a = swap_index[j] 
			b = swap_index[j+K]			
			m = np.random.randint(bottomx, W)

			# requires a deep copy
			_u_a = U[a,:,:, m:][:]
			_s_a = S[a,:, m:][:]
			_v_a = V[a,:, m:, :][:] 

			U[a,:,:, m:] = U[b,:,:, m:]
			S[a,:, m:] = S[b,:, m:]			
			V[a,:, m:, :] =	V[b,:, m:, :]		

			U[b,:,:, m:] = _u_a
			S[b,:, m:] = _s_a			
			V[b,:, m:, :] = _v_a

			data[a] = _reconstruction(U[a], S[a], V[a], mean, std)
			data[b] = _reconstruction(U[b], S[b], V[b], mean, std)

	else:
		assert False, "DatasetSVD() augment_method[{}] not supported".format(i)	

	return data

def preprocess_trainset():
	# Download testset	
	transform = transforms.Compose([transforms.ToTensor()])	
	test_dataset = datasets.CIFAR10('data/CIFAR10/', train=False, download=True, transform=transform)

	# Download trainset
	train_dataset = datasets.CIFAR10('data/CIFAR10/', train=True, download=False, transform=transform)

	# Pre-process trainset
	print("Decomposing CIFAR10 trainset for training...")
	print("This process is one-time and takes some time.")
	print("Result will be saved as data/cifair10_sp.train\n")	

	img, label = [], []
	U, S, V = [], [], []
	img_size = 32
	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=False)
	for idx, (data, target) in enumerate(dataloader):
		(b,c,w,h) = data.size()
		assert w==h and w==img_size, 'image size ({}, {}) does not match'.format(w,h)
		for j in range(b):
			im = data[j].numpy()
			label.append(target[j].item())	
			u, s, vh =svd(im, full_matrices=True)
			img.append(im)
			U.append(u)
			S.append(s)
			V.append(vh)
	np.save(decomposed_trainset, { "img": img, "label": label, "U":U, "S":S, "V":V }, allow_pickle=True)	

'''


