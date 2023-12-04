import numpy as np
from exp_smooth import _exp_smooth as es
import xarray as xr
import matplotlib.pyplot as plt

filename = "mi_upstream.nc"
ds = xr.open_dataset(filename, group="Reach_Timeseries")
# Get the first reach
data = np.ascontiguousarray(np.array(ds.Q.values)[:,0])
split_cutoff = data.shape[0] - 179

train_data = data[:split_cutoff]
test_data  = data[split_cutoff:]
model = es.UnivarSingleModel()

model.fit(train_data)
print(model)

predictions = model.predict_all(data)
loss = model.nrmse(data, split_cutoff)
nse  = model.nse(data, split_cutoff)
print("NRMSE loss: {loss}".format(loss=loss))
print("Nash-Sutcliffe: {nse}".format(nse=nse))

plt.plot(test_data)
plt.plot(predictions[split_cutoff:])
plt.show()    
