import pandas as pd
import numpy as np
import os,ndd
from embo import EmpiricalBottleneck

# Make into object
def get_sub_dat(fname,snum):
   # Load raw response data
    pdf = pd.read_csv('./data/'+fname)

    # Load session trial info
    seqID = pdf['Script_Picker'][0]
    pseq1 = pd.read_csv('./stimuli/low_hazard_HMM/low_hazard_HMM'+str(seqID)+'.csv')
    pseq2 = pd.read_csv('./stimuli/low_hazard_MM/low_hazard_MM'+str(seqID)+'.csv')
    pseq3 = pd.read_csv('./stimuli/high_hazard_HMM/high_hazard_HMM'+str(seqID)+'.csv')
    pseq4 = pd.read_csv('./stimuli/high_hazard_MM/high_hazard_MM'+str(seqID)+'.csv')
    seqDict = {'HMM_low':pseq1,'MM_low':pseq2,'HMM_high':pseq3,'MM_high':pseq4}

    # Add response code to match data sequence file
    stim_im = ["Images/NewTrialBC.png","Images/NewTrialWC.png","Images/NoChoice.png", 
               "Images/NoChoiceMM.png", "Images/NewTrialWCMM.png", "Images/NewTrialBCMM.png"]
    pdf['resp']=np.nan
    pdf['resp'].loc[(pdf['key_press'] == 37) & (pdf['stimulus'].isin(stim_im))] = 1
    pdf['resp'].loc[(pdf['key_press'] == 39) & (pdf['stimulus'].isin(stim_im))] = 2

    # Add bead info
    pdf['bead'] = np.nan
    offset = np.array(pdf['stimulus'])
    beads = np.zeros((len(offset)))-1
    beads[np.isin(offset,["Images/BlackBeadChoiceCorrect.png","Images/WhiteBeadChoiceWrong.png",
                         "Images/BlackBeadChoiceCorrectMM.png","Images/WhiteBeadChoiceWrongMM.png"])] = 1
    beads[np.isin(offset,["Images/WhiteBeadChoiceCorrect.png","Images/BlackBeadChoiceWrong.png",
                         "Images/WhiteBeadChoiceCorrectMM.png","Images/BlackBeadChoiceWrongMM.png"])] = 2
    pdf['bead'].iloc[:-2] = beads[2:]

    # Add session number and condition
    # Get indicies of start and finish of each block
    pdf['session_num'] = np.nan
    pdf['condition'] = np.nan
    sess_order = ['HMM_low','MM_low','HMM_high','MM_high']
    bidx = []
    # Get all indicies where key_press is equal to 32 - helps demarcate block bounds
    p32 = pdf[pdf['key_press'] == 32]
    for i in np.arange(4):
        if i == 0:
            sidx = (p32.index[p32.index < 100].max(), p32.index[(p32.index > 100)].min())
        else:
            prevIdx = bidx[i-1][1]
            start = p32.index[p32.index > prevIdx].min()
            end = p32.index[p32.index > start].min()
            sidx = (start,end)
        bidx.append(sidx)

        # Add session info
        pdf['session_num'].loc[pdf.index.isin(np.arange(sidx[0]+1,sidx[1]))] = i+1
        pdf['condition'].loc[pdf.index.isin(np.arange(sidx[0]+1,sidx[1]))] = sess_order[i]

    # Get jar info
    jar = np.concatenate((np.array(pseq1['Urn']),
                         np.array(pseq2['Urn']),
                         np.array(pseq3['Urn']),
                         np.array(pseq4['Urn'])))
    
    # Get clean data and check that bead sequences match per session
    # Also add trial since jar changepoint
    pdf_clean = pdf[pdf['resp'].notna()].copy()
    pdf_clean['TSCP'] = np.nan
    pdf_clean['TSCP2'] = np.nan
    for i,sess in enumerate(sess_order):
        seqBeads = np.array(seqDict[sess]['Bead'])
        rawBeads = np.array(pdf_clean['bead'].loc[pdf_clean['condition'] == sess])
        if sum(seqBeads!=rawBeads)>0:
            print('Data not aligned for subject %str'%str(snum))

        # Get trial since changepoint
        tscp = 0
        currJar = 0
        tscp_vc = np.zeros(len(seqBeads))
        for i,j in enumerate(seqDict[sess]['Urn']):
            if i == 0:
                currJar = j
            else:
                if currJar !=j:
                    tscp = 0
                    currJar = j
                else:
                    tscp += 1
            tscp_vc[i] = tscp
        pdf_clean['TSCP'].loc[pdf_clean['condition'] == sess] = tscp_vc
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP'] == 0] = '0'
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP'] == 1] = '1'
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP'] == 2] = '2'
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP'].isin([3,4])] = '3-4'
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP'].isin([5,6,7])] = '5-7'
        pdf_clean['TSCP2'].loc[pdf_clean['TSCP']>7] = '8+'

    # Create new data frame with columns of interest
    df = pd.DataFrame({
        "Subject": np.array([snum]*len(jar)),
        "Sequence": np.array([seqID]*len(jar)),
        "Session": pdf_clean['condition'],
        "SessTrial": np.concatenate((np.arange(1,201),np.arange(1,201),np.arange(1,201),np.arange(1,201))),
        "TSCP": pdf_clean['TSCP'],
        "TSCP2": pdf_clean['TSCP2'],
        "Jar":jar,
        "Bead": pdf_clean['bead'],
        "Resp": pdf_clean['resp'],
        "Correct_Bead": np.array(np.array(pdf_clean['resp']) == np.array(pdf_clean['bead'])).astype(int), 
        "Correct_Jar": np.array(np.array(pdf_clean['resp']) == jar).astype(int),
        "RT": pdf_clean['rt']
    })

    return(df) 


# For each subject, compute MI as a function of window size
def get_windowed_x(x,z,w=1):
    zw = z[w:].copy()
    aux_base = 2**np.arange(w)
    xw_binned = np.array([sum(aux_base*np.array(x[i:(i+w)])) for i in np.arange(len(zw))])
    return(xw_binned,zw)

def get_XY(df,m,sem,sess):
    
    x = np.array(df['TSCP2'].loc[df['Session'] == sess])
    y = np.array(df[m].loc[df['Session'] == sess])
    yerr = np.array(df[sem].loc[df['Session'] == sess])
    
    return(x,y,yerr)
    
def mutual_inf_nsb(x,y,ks):
    """
    Calculate mutual information using NSB method
    """
    ar = np.column_stack((x,y))
    mi = ndd.mutual_information(ar,ks)
    return np.log2(np.e)*mi #ndd returns nats - multiply by log2(e) to convert to bits