from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

import torch
from tqdm import tqdm

class Utils:
    def get_magnitudes(vec_array=np.array([])):
        '''
            Get magnitudes for all vectors from vector array
        '''
        magn = []
        for i in range(vec_array.shape[0]):
            magn.append(np.sqrt(vec_array[i].dot(vec_array[i])))
        return np.array(magn)
    
    def plot_for_time_interval(xdata, ydata, t_start=0, t_end=100000, xdata2=np.array([]), ydata2=np.array([]),
                            xdata3=np.array([]), ydata3=np.array([]), fsize=(12,9)):
        '''
            xdata supposed to be in nanoseconds
        '''
        figure(figsize=fsize, dpi=80)
        xdatas = xdata * 1e-9
        xval = (xdatas - xdatas[0])[((xdatas - xdatas[0]) <= t_end) & ((xdatas - xdatas[0]) >= t_start)]
        yval = ydata[((xdatas - xdatas[0]) <= t_end) & ((xdatas - xdatas[0]) >= t_start)]
        plt.plot(xval, yval)
        if xdata2.size > 0:
            xdatas2 = xdata2 * 1e-9
            xval2 = (xdatas2 - xdatas2[0])[((xdatas2 - xdatas2[0]) <= t_end) & ((xdatas2 - xdatas2[0]) >= t_start)]
            yval2 = ydata2[((xdatas2 - xdatas2[0]) <= t_end) & ((xdatas2 - xdatas2[0]) >= t_start)]
            plt.plot(xval2, yval2)
        if xdata3.size > 0:
            xdatas3 = xdata3 * 1e-9
            xval3 = (xdatas3 - xdatas3[0])[((xdatas3 - xdatas3[0]) <= t_end) & ((xdatas3 - xdatas3[0]) >= t_start)]
            yval3 = ydata3[((xdatas3 - xdatas3[0]) <= t_end) & ((xdatas3 - xdatas3[0]) >= t_start)]
            plt.plot(xval3, yval3)
            
    def make_labeling_for_tick_interval(labels, ind_start=0, ind_end=0, label=-1):
        '''
            Set labels for interval from ind_start to ind_end.
            ind_end frame is not included.
            If ind_end < 0 then it set label to the last label.
        '''
        if (len(labels) < 1):
            print("Labels array is empty")
            return []
        endtick = (ind_end if ind_end > 0 else len(labels))
        for i in range(ind_start, endtick):
            labels[i] = label
        return labels
    
    def make_labeling_for_time_interval(labels, timestamp, time_start=0, time_end=0, label=-1):
        '''
            Set labels for time interval from time_start to time_end.
            time_end frame is not included.
            If time_end < 0 then it set label to the last label.
        '''
        if (len(labels) < 1):
            print("Labels array is empty")
            return []
        
        i_start = 0
        i_end = -1
        
        i = 0
        tmp = timestamp - timestamp[0]
        tmp *= 1e-9
        while tmp[i] < time_start:
            i += 1
        i_start = i
        if time_end > 0:
            while tmp[i] < time_end:
                i += 1
            i_end = i
        
        return Utils.make_labeling_for_tick_interval(labels, i_start, i_end, label)

    def make_transformations(data):
        # For counting the time from moment of maneuver start
        _means = np.zeros(len(data[0]), dtype=float)
        _means[0] = data[0,0]
        data -= _means
         
        _char_vals = np.ones(len(data[0]), dtype=float)
        _char_vals[0] = 1e9
        _char_vals[4] = 9.8
        _char_vals[5] = 9.8
        _char_vals[6] = 9.8
        # _char_vals = [1e9, 1, 1, 1, 9.8, 9.8, 9.8]
        
        data /= _char_vals
        
        # add magnitudes for velocity and acceleration
        data = np.append(data, np.linalg.norm([data[:, 1:4]], axis = 2).transpose(), axis = 1)
        data = np.append(data, np.linalg.norm([data[:, 4:7]], axis = 2).transpose(), axis = 1)
        
        return data
    
    
def replace_idx_to(_data, old_idx, new_idx):
    _vvv = _data
    for i in np.where(_data == old_idx):
        _vvv[i] = new_idx
    return _vvv    
    
    
def validate(model, _testloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for obj, labels in _testloader:
            outputs = model(obj.to(device))
            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total 

def predict(model, X, device='cpu'):
    with torch.no_grad():
        outputs = model(X.to(device))
        _, predicted = torch.max(outputs.data.cpu(), 1)
    return predicted

def predict_for_dataloader(model, _loader, device='cpu'):
    all_predicted = torch.tensor([])
    all_labels = torch.tensor([])
    with torch.no_grad():
        for objs, labels in _loader: # get batch
            outputs = model(objs.to(device))
            _, predicted = torch.max(outputs.data.cpu(), 1)
            all_predicted = torch.cat((all_predicted, predicted), axis=0)
            all_labels = torch.cat((all_labels, torch.tensor(labels)), axis=0)
    return all_predicted, all_labels

def train(model, criterion, optimizer, _trainloader, 
          _valloader, epochs=10, device='cpu'):
    for epoch in range(epochs):
        for objs, labels in _trainloader: # get batch
            output = model(objs.to(device))
            
            loss = criterion(output.to(device), labels.to(device))
            loss.backward() # backprop
            optimizer.step()    # Does the update
            optimizer.zero_grad()   # zero the gradient buffers
            
    accuracy = validate(model, _valloader, device=device)
    return accuracy


def calc_scores(_y_pred, _y_real):
    _a = accuracy_score(_y_real, _y_pred)
    _pr = precision_score(_y_real, _y_pred, average='weighted')
    _re = recall_score(_y_real, _y_pred, average='weighted')
    _f1 = f1_score(_y_real, _y_pred, average='weighted')
    _ja = jaccard_score(_y_real, _y_pred, average='weighted')
    _ja_m = np.min(jaccard_score(_y_real, _y_pred, average=None))
    return _a, _pr, _re, _f1, _ja, _ja_m