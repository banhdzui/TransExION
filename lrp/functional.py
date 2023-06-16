'''
Created on 24 Feb 2021

@author: danhbuithi
'''
import numpy as np 
import matplotlib.pyplot as plt

import torch
 
from lrp.explanator import LRPGenerator

def import_regression_result(output, target, y_pred_scores, y_trues):
    predicted_scores = output.tolist()
    y_pred_scores.extend(predicted_scores)
    true_labels = target.tolist()
    y_trues.extend(true_labels)
    
def train_spectral_similarity_measure(epoch, niter, data_loader, net, optimizer, criterion, device, logger_file):
    train_losses = []
    net.train()
    for i, data in enumerate(data_loader, 0):
        mz_f, mz_mask, loss_f, loss_mask, target = data 
        
        mz_f = mz_f.to(device)
        mz_mask = mz_mask.to(device)
        
        loss_f = loss_f.to(device)
        loss_mask = loss_mask.to(device)
        
        output = net(mz_f, mz_mask, loss_f, loss_mask) 
    
        output = output.view(-1)
        target = target.to(device)
        
        loss = criterion(output, target)
        train_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % int(len(data_loader) / 10 + 1) == 0:# and opt.verbal:
            message = '[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(data_loader), loss.data) 
            logger_file.info(message)
    logger_file.info('avg ' + str(np.average(train_losses)))
    
def test_spectral_similarity_measure(data_loader, net, criterion, device):
    val_losses = []
    net.eval()
    for i, data in enumerate(data_loader, 0):      
        
        mz_f, mz_mask, loss_f, loss_mask, target = data
        
        mz_f = mz_f.to(device)
        mz_mask = mz_mask.to(device)
        loss_f = loss_f.to(device)
        loss_mask = loss_mask.to(device)
        
        output = net(mz_f, mz_mask, loss_f, loss_mask)  

        output = output.view(-1)
        target = target.to(device)
        
        loss = criterion(output, target)
        val_losses.append(loss.item())
        
        if i % int(len(data_loader) / 10 + 1) == 0:# and opt.verbal:
            message = '[%d/%d] Loss: %.4f' % (i, len(data_loader), loss.data) 
            print(message)
    return np.average(val_losses)


def evaluate_spectral_similarity_measure(data_loader, net, device):
    predict_values = []
    true_values = []
    
    net.eval()
    for i, data in enumerate(data_loader, 0):      
        mz_f, mz_mask, loss_f, loss_mask, target = data
        
        mz_f = mz_f.to(device)
        mz_mask = mz_mask.to(device)
        loss_f = loss_f.to(device)
        loss_mask = loss_mask.to(device)
        
        output = net(mz_f, mz_mask, loss_f, loss_mask) 
    
        output = output.view(-1)
        target = target.to(device)
        import_regression_result(output, target, predict_values, true_values)
        
        if i % int(len(data_loader) / 10 + 1) == 0:# and opt.verbal:
            message = 'evaluating [%d/%d]' % (i, len(data_loader)) 
            print(message)
            
    return np.array(predict_values), np.array(true_values)

def _find_mz(x, mz):
    d = np.fabs(x - mz)
    i = np.argmin(d)
    if d[i] < 1.0: return mz[i]
    return None 
     

def draw_explaining_heatmap(mz, check_mz, fragment_weights, heat_map, diff_map, topk=20):
    '''
    fragment_weights: L 
    heat_map: L x 3 
    '''
    
    L = min(topk, heat_map.shape[0])
    fragment_weights = fragment_weights.numpy()[1:]
    sorted_fragments = np.argsort(fragment_weights)[::-1][:L]
    sorted_fragments = np.sort(sorted_fragments)    
    
    reduced_heatmap = np.zeros((L, heat_map.shape[1]))
    for i, y in enumerate(sorted_fragments): 
        reduced_heatmap[i] = heat_map[y] * fragment_weights[y] 
        
    plt.figure(figsize=(10, L*0.25))
    color_map = plt.pcolor(reduced_heatmap, cmap='YlOrBr')
    for i, y in enumerate(sorted_fragments):
        for x in range(heat_map.shape[1]):
            if heat_map[y, x] <= 0 or fragment_weights[y] <= 0: continue
            
            lower_bound = mz[y]-diff_map[y, x]
            d_lower = _find_mz(lower_bound, check_mz)
            
            upper_bound = mz[y] + diff_map[y, x]
            d_upper = _find_mz(upper_bound, check_mz)
            
            e = ''
            if d_lower is not None and d_lower <= mz[y]: e = '%.2f' % d_lower 
            else: e = '-' 
            
            e += ', (%.2f, %.2f), ' % (mz[y], diff_map[y, x])
            if d_upper is not None and d_upper > mz[y]: e += '%.2f' % upper_bound 
            else: e += '-'
            
            plt.text(x + 0.5, i + 0.5, e,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize='small')

    plt.colorbar(color_map)
    
             
                
def compute_explaining_heatmap(max_len, R, k=3):
    R, indices = torch.max(R, -1) #N x L
    sorted_R = torch.argsort(R, dim=-1, descending=True)
    L, _ = R.shape
    L = L - 1 # remove the fake fragment
    L = min(max_len, L)
    heat_map = np.zeros((L, k))
    diff_map = np.zeros((L, k))
    for i in range(1, L+1): 
        for j in range(k):
            int_mass = sorted_R[i, j]
            defect_mass = max(0, indices[i, int_mass] - 2)
            heat_map[i-1,j] = R[i, int_mass]
            diff_map[i-1,j] = max(int_mass-1, 0) + defect_mass/100.0
    return heat_map, diff_map
    
    
def draw_spectral_pair(query_mz, query_intensity, ref_mz, ref_intensity):
    _, axs = plt.subplots(2, sharex=True)
    axs[0].bar(query_mz, query_intensity, 0.25, color='red')
    axs[1].bar(ref_mz, ref_intensity, 0.25, color='red')
               
    for i, v in enumerate(query_mz):
        #if intensity1[i] < 0.05: continue 
        axs[0].text(v, query_intensity[i] + .05, str(round(v, 3)), color='blue', fontsize='small')
        
    for i, v in enumerate(ref_mz):
        #if intensity2[i] < 0.05: continue
        axs[1].text(v, ref_intensity[i] + .05, str(round(v, 3)), color='blue', fontsize='small')


def group_spectra_by_queryid(data_set):
    pairs = data_set.get_short_info_all_pairs()
    pair_dict = {}
    for i, v in enumerate(pairs):
        q, _, score = v
        if q not in pair_dict: 
            pair_dict[q] = []
        pair_dict[q].append((i, score))
    return pair_dict 
        
    
def explain_spectral_similarity(data_set, net, device, output_folder):
    net.eval()
    lrp_generator = LRPGenerator(net)
    
    pairs = data_set.get_short_info_all_pairs()
    for j, _ in enumerate(pairs):
        mz_f, mz_mask, loss_f, loss_mask, u, v, _ = data_set.get_full_info(j)
        
        mz_f = torch.unsqueeze(mz_f, 0).to(device)
        mz_mask = torch.unsqueeze(mz_mask, 0).to(device)
        loss_f = torch.unsqueeze(loss_f, 0).to(device)
        loss_mask = torch.unsqueeze(loss_mask, 0).to(device)
          
        score, full_mz_R, fragment_weights = lrp_generator.generate(mz_f, mz_mask, loss_f, loss_mask)
        print(score)
        if score < 0.6: continue 
        full_mz_R = full_mz_R[0]
        
        mz_len = len(u[2])
        mz_heatmap, mz_diffmap = compute_explaining_heatmap(mz_len, full_mz_R, k=3)
        draw_explaining_heatmap(u[2], v[2], fragment_weights[0], mz_heatmap, mz_diffmap, topk=20)
        file_name = '_'.join([str(u[0]), str(v[0]), u[1], v[1]]) 
        plt.savefig(output_folder + '/' + file_name + '_00.jpg')
        plt.close()
            
