import os
import numpy as np
from prettytable import PrettyTable
from svrandom import seed, Blindtest

oodd_fpr_levels = [0,10,15,20]

seed()
tester = Blindtest()

for fpr_level in oodd_fpr_levels:

	result_dict = tester.show_testresult(fpr_level)	
	tb_header = {}
	claerror_average, tpr_average = {}, {}

	for classifier, result in result_dict.items():
		tb_header = [] 
		for mt, (tpr, claerror) in result.items():
			tb_header.append(mt)
			if classifier not in claerror_average.keys():
				tpr_average[classifier] = [tpr]
				claerror_average[classifier] = [claerror]
			else:
				tpr_average[classifier].append(tpr)
				claerror_average[classifier].append(claerror)
	

	if fpr_level == 0:
		# Classification error
		print("\n")
		table = PrettyTable( )
		table.field_names = [""] + [tester.methods[mt] for mt in tb_header]
		for k in tpr_average.keys():
			table.add_row([k] + ["%0.3f" % claerror_average[k][i] for i in range(len(tpr_average[k]))])
		print("Classification Error (Adversarial Defense Only)".format(fpr_level))
		print(table)	

	else:
		# TPR of OOD detection		  
		table = PrettyTable( )
		table.field_names = [""] + [tester.methods[mt] for mt in tb_header]
		for k in tpr_average.keys():
			table.add_row([k] + ["%0.3f" % tpr_average[k][i] for i in range(len(tpr_average[k]))])
		print("OOD detection TPR @ {}% FPR of ID".format(fpr_level))
		print(table)

		# Classification error
		table = PrettyTable()
		table.field_names = [""] + [tester.methods[mt] for mt in tb_header]
		for k in tpr_average.keys():
			table.add_row([k] + ["%0.3f" % claerror_average[k][i] for i in range(len(tpr_average[k]))])
		print("Classification Error")
		print(table)
	

	print("\n")