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

predictions = model.predict_all(data)
test_loss = model.nrmse(data, split_cutoff)
train_loss = model.nrmse(train_data, 0)
after_200_loss = model.nrmse(train_data, 200)
after_300_loss = model.nrmse(train_data, 300)
nse  = model.nse(data, split_cutoff)

print("In-sample NRMSE loss: {loss}".format(loss=train_loss))
print("Out-of-sample NRMSE loss: {loss}".format(loss=test_loss))
print("Nash-Sutcliffe efficiency: {nse}".format(nse=nse))
print()
print("In-sample NRMSE loss after day 200: {loss}".format(loss=after_200_loss))
print("In-sample NRMSE loss after day 300: {loss}".format(loss=after_300_loss))

plt.plot(data)
plt.plot(np.arange(split_cutoff, data.shape[0] + 1, step=1), predictions[split_cutoff:])
plt.legend(["Observed Data", "Prediction"])
plt.tight_layout()
plt.show()
