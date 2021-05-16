#import necessary libraries
from sklearn.metrics import r2_score
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error,mean_absolute_error


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#calculate the predicted scores and print it
def scores(actuall_labels, predicted_labels):
	print('Pearson Score: ',pearsonr(actuall_labels.to_numpy().reshape(-1,),predicted_labels.reshape(-1,))[0])
	print('R2_Score: ',r2_score(actuall_labels,predicted_labels))
	print('Spearmanr Score: ',spearmanr(actuall_labels.to_numpy().reshape(-1,),predicted_labels.reshape(-1,))[0])
	print('Mean Squared Error(MSE): ',mean_squared_error(actuall_labels,predicted_labels))
	print('Mean Absolute Error(MAE): ',mean_absolute_error(actuall_labels,predicted_labels))


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#save the predicted values with their IDs
def saveOutput(data_ids, predicted_labels, file_name='predicted.txt'):
	if len(data_ids) != len(predicted_labels):
		print('Length Mismatch Error!')
		return
	
	with open(file_name,'w+') as outfile:
		for i in range(len(predicted_labels)):
			outfile.write(data_ids[i]+','+str(predicted_labels[i])+'\n')


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------



