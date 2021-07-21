#!/usr/bin/env python
# coding: utf-8

# # Helper functions for Information Bottleneck with Confidence

# In[13]:


# !jupyter nbconvert --to script smfile.ipynb


# In[1]:


import pandas as pd
import numpy as np
from embo import InformationBottleneck
import HMM_beads_utils as ut
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import os,ndd,pickle
from adjustText import adjust_text
cmap = plt.get_cmap("tab10")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


def get_previous_n_beads(w, x):
    '''
    Function that gets history of viewed beads
    Input:
        - w: number of past beads in history
        - x: full sequence of beads
    Output:
        - history: array of strings of previous w beads
    '''
    numtrials = x.shape[0]
    aux_base = 2**np.arange(w)
    x_padded = np.empty(numtrials+w)
    x_padded[:w] = 0
    x_padded[w:] = x
    history_bin = np.array([sum(aux_base * np.array(x_padded[i:(i + w)][::-1])) for i in np.arange(numtrials)])
    return [np.binary_repr(int(z),w+1) for z in history_bin]


# In[3]:


def get_windowed_x_v(xhist,z,w=1):
    zw = z[w:].copy()
    xw = np.array([int(h[1:][::-1],2) for h in xhist][w:])
    return(xw,zw)


# In[4]:


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


# In[5]:


def gen_sim_pred(sname,beads,jars,trials, p = 0.8,h = 0.01):
    '''
    Function to generate responses from the ideal observer with different internal estimates of p and h
    INPUT:
        - sname: subject identifier in data frame
        - beads: 1D array of observed bead sequence
        - jars: 1D array of jar sequence that generated beads
        - p: internal model estimate of the probability that the bead will match the generating jar
        - h_array: internal estimate of the hazard rate
    OUTPUT:
        -df: data frame with the following columns
            - Subject: subject ID + simulation type (from no noise to lots of noise)
            - Trial: trial number from start of experiment
            - Jar: jar generating the bead
            - Bead: bead observed on each trial
            - Prior: prior probability that the the jar generating the next bead = 1 (before observing the bead)
            - Posterior: posterior probability that the jar generating the last bead = 1 (after observing the bead)
            - Prediction: simulated prediction
    '''
    
    # Initialize
    p_jars = np.array([.5,.5])              # Initial prior
    likes = np.array([[p,(1-p)],[(1-p),p]]) # Likelihood
    prior = np.zeros(len(beads))            # Initialize trial-by-trial priors
    pred_hardmax = np.zeros(len(beads))     # Initialize hardmax predictions
    posterior = np.zeros(len(beads))        # Initialize trial-by-trial posteriors
    
    # Loop through all trials and get simulated responses
    for i in np.arange(len(beads)):
        # Apply hazard rate
        #h = h_array[i]
        z1,z2 = tuple(p_jars.copy())
        p_jars = [(1-h)*z1 + h*z2, (1-h)*z2 + h*z1]

        #Save Prior
        prior[i] = p_jars[1]
        
        # Incorporate likelihood
        p_jars = p_jars*likes[int(beads[i])]
        p_jars = p_jars/sum(p_jars)
        
        # Save posterios
        posterior[i] = p_jars[1]
        
    # Get harmax predictions
    pred_hardmax[prior!=.5] = np.array(prior[prior!=.5] > .5).astype(int)
    pred_hardmax[prior==.5] = np.random.choice([0,1],size=len(prior[prior==.5]),replace=True)
    
    # Save data frame
    df = pd.DataFrame({
        'Subject':np.array([str(sname)+'_Ideal']*len(beads)),
        'Trial':trials,
        'Jar':jars,
        'Bead':beads,
        'Prior':prior,
        'Posterior':posterior,
        'Prediction':pred_hardmax
    })
    return(df)


# In[6]:


# Test bounds in each session
def get_windowed_mi(x_array,z,nw,confid = 0):
    '''
    Function that windows of observations from a sequence x offset from y and computes the MI
    Input:
        - x: sequence of "past" observations to be windowed
        - z: sequence that serves as the "future" events
        - nw: maximum window size
    Output:
        - mis: NSB estimated mutual informtion values for each window size
    '''
    mis = np.zeros(nw+1) #vector of mutual informations for different window sizes
    for w in np.arange(1,nw+1):
        xw,zw = get_windowed_x_v(['0' + h[-w:] for h in x_array],z,w=w)         # use convenience function from utilities to window x
        mis[w] = ut.mutual_inf_nsb(xw,zw,[2**w,2+confid*2]) # use convenience function from utilities to compute NSB mutual info
    return(mis)


# In[7]:


def get_windowed_bound(x_array,z,w,mb=50,nb=2000,p=8):
    '''
    Function that computes the information bottleneck bound between two discrete variables
    Input:
        - x: sequence of past events
        - z: sequence of future events
        - w: size of the window to be applied to x
        - mb: maximum value of the lagrange multiplier beta
        - nb: number of values of beta between 0 and mb to be run through the Blahaut-Arimoto algorithm
        - p: number of cores to use in the computation (to help with speed)
    Output:
        - 4 element tuple with a sequence of ipast and ifuture values corresponding to the information bottleneck for full and one back bounds
    '''
    # Get desired windowed x and corresponding z
    xw,zw = get_windowed_x_v(['0' + h[-w:] for h in x_array],z,w=w)
    
    # Compute the information bottleneck using EMBO for the window size w specified above
    ipw, ifw, _, _= InformationBottleneck(xw, zw, window_size_x=1, window_size_y=1).get_bottleneck()

    # Compute the information bottleneck using EMBO for the window size w specified above
    x1b,z1b = get_windowed_x_v(['0' + h[-1:] for h in x_array],z,w=1)
    ip1b, if1b, _, _ = InformationBottleneck(x1b, z1b, window_size_x=1, window_size_y=1).get_bottleneck()   
    
    # Return informtion bottleneck
    return((ipw,ifw,ip1b,if1b))


# In[8]:


def get_windowed_mi_subs(sdat_all,nw):
    '''
    Function to compute windowed mutual information values for all subjects
    Input:
        - sdat_all: data frame with all of the subject tones, responses, and sources
        - nw: maximum window size to be considered
    Output:
        - Ipast for each window size and Ifuture
    '''
    # Create dictionaries to keep track of variables with subject IDs as the keys
    subs = pd.unique(sdat_all['Subject'])
    ip_sub = {}
    if_sub = {}
    # Loop through each subject and get their Ipast for windows up to nw and their ifuture
    for subi,sub in enumerate(subs):
        sdat = sdat_all[sdat_all['Subject'] == sub]                   # Get data from specific subject 
        ip_sub[sub] = get_windowed_mi(sdat['History'],sdat['Response'],nw)    # compute Ipast using function defined above 
        if_sub[sub] = ut.mutual_inf_nsb(sdat['Response'],sdat['Jar'],[2,2]) # compute Ifuture
    # Return dictionaries of ipast and ifuture values
    return(ip_sub,if_sub)


# In[9]:


def get_bootstrapped_samples(dat,wips,nboot):
    '''
    Function to compute and return bootstrapped distributions of Ipast and Ifuture values for each subject
    Input:
        - dat: all subject data
        - wips: matrix of mutual information by widow size values for each subject
        - nboot: number of requested bootstrap iterations
    Output:
        - dictionaries of bootstrapped distributions of Ipast and Ifuture values
    '''
    # Get subject IDs and initialize dictionaries
    subs = pd.unique(dat['Subject'])
    ip_boot_mi = {}
    if_boot_mi = {}
    
    # Loop through each subject and get bootstrapped estimates
    for subi,sub in enumerate(subs):
        sdat = dat[dat['Subject'] == sub]   # Get subject data
        w = wips[sub].argmax()              # Get subject's maximum window size
        x = np.array(sdat['History'])           # Get tones, responses, and sources for the subject
        r = np.array(sdat['Response'])
        z = np.array(sdat['Jar'])
        xw,rw = get_windowed_x_v(['0' + h[-(w+1):] for h in x],r,w=(w+1)) # Window the tones and responses appropriately
        ip_boot_mi[sub] = np.zeros(nboot)      # Initialize arrays of mutual information values
        if_boot_mi[sub] = np.zeros(nboot)
        
        # Run bootstrap procedure
        for boot in np.arange(nboot):
            idx = np.random.choice(np.arange(len(xw)),size=len(xw),replace=True)    # Select random indicies with replacement
            ip_boot_mi[sub][boot] = ut.mutual_inf_nsb(xw[idx],rw[idx],[2**(w+1),2]) # Use indexed data to compute Ipast
            if_boot_mi[sub][boot] = ut.mutual_inf_nsb(r[idx],z[idx],[2,2])          # Use indexed data to compute Ifuture
    # Return dictionaries of Ipast and Ifuture distributions
    return(ip_boot_mi,if_boot_mi)


# In[10]:


def get_windowed_mi_subs_confid(sdat_all,nw):
    '''
    Function to compute windowed mutual information values for all subjects
    Input:
        - sdat_all: data frame with all of the subject tones, responses, and sources
        - nw: maximum window size to be considered
    Output:
        - Ipast for each window size and Ifuture
    '''
    # Create dictionaries to keep track of variables with subject IDs as the keys
    subs = pd.unique(sdat_all['Subject'])
    ip_sub = {}
    if_sub = {}    
    # Loop through each subject and get their Ipast for windows up to nw and their ifuture
    for subi,sub in enumerate(subs):
        sdat = sdat_all[sdat_all['Subject'] == sub]                   # Get data from specific subject 
        ip_sub[sub] = get_windowed_mi(sdat['History'],2*sdat['Response'] + sdat['Confidence'],nw, confid = 1)    # compute Ipast using function defined above 
        if_sub[sub] = ut.mutual_inf_nsb(2*sdat['Response'] + sdat['Confidence'],sdat['Jar'],[4,2]) # compute Ifuture
    # Return dictionaries of ipast and ifuture values
    return(ip_sub,if_sub)



def get_bootstrapped_samples_confid(dat,wips,nboot):
    '''
    Function to compute and return bootstrapped distributions of Ipast and Ifuture values for each subject
    Input:
        - dat: all subject data
        - wips: matrix of mutual information by widow size values for each subject
        - nboot: number of requested bootstrap iterations
    Output:
        - dictionaries of bootstrapped distributions of Ipast and Ifuture values
    '''
    # Get subject IDs and initialize dictionaries
    subs = pd.unique(dat['Subject'])
    ip_boot_mi = {}
    if_boot_mi = {}
    
    # Loop through each subject and get bootstrapped estimates
    for subi,sub in enumerate(subs):
        sdat = dat[dat['Subject'] == sub]   # Get subject data
        w = wips[sub].argmax()              # Get subject's maximum window size
        x = np.array(sdat['History'])           # Get tones, responses, and sources for the subject
        r = np.array(2*sdat['Response'] + sdat['Confidence'])
        z = np.array(sdat['Jar'])
        xw,rw = get_windowed_x_v(['0' + h[-(w+1):] for h in x],r,w=(w+1)) # Window the tones and responses appropriately
        ip_boot_mi[sub] = np.zeros(nboot)      # Initialize arrays of mutual information values
        if_boot_mi[sub] = np.zeros(nboot)
            
        # Run bootstrap procedure
        for boot in np.arange(nboot):
            idx = np.random.choice(np.arange(len(xw)),size=len(xw),replace=True)    # Select random indicies with replacement
            ip_boot_mi[sub][boot] = ut.mutual_inf_nsb(xw[idx],rw[idx],[2**(w+1),4]) # Use indexed data to compute Ipast
            if_boot_mi[sub][boot] = ut.mutual_inf_nsb(r[idx],z[idx],[4,2])
    # Return dictionaries of Ipast and Ifuture distributions
    return(ip_boot_mi,if_boot_mi)


# In[11]:


def get_windowed_mi_subs_confo(sdat_all,nw):
    '''
    Function to compute windowed mutual information values for all subjects
    Input:
        - sdat_all: data frame with all of the subject tones, responses, and sources
        - nw: maximum window size to be considered
    Output:
        - Ipast for each window size and Ifuture
    '''
    # Create dictionaries to keep track of variables with subject IDs as the keys
    subs = pd.unique(sdat_all['Subject'])
    ip_sub = {}
    if_sub = {}
    # Loop through each subject and get their Ipast for windows up to nw and their ifuture
    for subi,sub in enumerate(subs):
        sdat = sdat_all[sdat_all['Subject'] == sub]                   # Get data from specific subject 
        ip_sub[sub] = get_windowed_mi(sdat['History'],sdat['Confidence'],nw)    # compute Ipast using function defined above 
        if_sub[sub] = ut.mutual_inf_nsb(sdat['Confidence'],sdat['Jar'],[2,2]) # compute Ifuture
    # Return dictionaries of ipast and ifuture values
    return(ip_sub,if_sub)

def get_bootstrapped_samples_confo(dat,wips,nboot):
    '''
    Function to compute and return bootstrapped distributions of Ipast and Ifuture values for each subject
    Input:
        - dat: all subject data
        - wips: matrix of mutual information by widow size values for each subject
        - nboot: number of requested bootstrap iterations
    Output:
        - dictionaries of bootstrapped distributions of Ipast and Ifuture values
    '''
    # Get subject IDs and initialize dictionaries
    subs = pd.unique(dat['Subject'])
    ip_boot_mi = {}
    if_boot_mi = {}
    
    # Loop through each subject and get bootstrapped estimates
    for subi,sub in enumerate(subs):
        sdat = dat[dat['Subject'] == sub]   # Get subject data
        w = wips[sub].argmax()              # Get subject's maximum window size
        x = np.array(sdat['History'])           # Get tones, responses, and sources for the subject
        r = np.array(sdat['Confidence'])
        z = np.array(sdat['Jar'])
        xw,rw = get_windowed_x_v(['0' + h[-(w+1):] for h in x],r,w=(w+1)) # Window the tones and responses appropriately
        ip_boot_mi[sub] = np.zeros(nboot)      # Initialize arrays of mutual information values
        if_boot_mi[sub] = np.zeros(nboot)
            
        # Run bootstrap procedure
        for boot in np.arange(nboot):
            idx = np.random.choice(np.arange(len(xw)),size=len(xw),replace=True)    # Select random indicies with replacement
            ip_boot_mi[sub][boot] = ut.mutual_inf_nsb(xw[idx],rw[idx],[2**(w+1),2]) # Use indexed data to compute Ipast
            if_boot_mi[sub][boot] = ut.mutual_inf_nsb(r[idx],z[idx],[2,2])
    # Return dictionaries of Ipast and Ifuture distributions
    return(ip_boot_mi,if_boot_mi)


# In[ ]:


def add_noise(x, level):
    n = 1
    #np.random.seed(17)
    noise = np.random.binomial(n, 1-level, len(x))
    return(np.where(x == noise,1,0))

