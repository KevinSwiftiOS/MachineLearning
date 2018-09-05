import numpy as np;
arr = np.array([[1,2,3,4],[5,6,7,8]]);
print(arr);
print(arr - np.max(arr,axis=1).reshape(arr.shape[0],1));
print(arr.shape[1]);
#行方向求和
exp_sum = (np.sum(np.exp(arr), axis=1)).reshape(arr.shape[0], 1);
print(exp_sum);
print(np.exp(arr) / exp_sum);