import os 
import copy
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

def Organize(data_file:str, n_rows:int, n_columns:int, total_run_time:float, sampling_rate:float, debug = False)->dict:
    '''
    Creates a 3D array where the first dimension is represents the plate rows, the second represents the plate columns, and the third contains the timeseries data.
    This is meant to be used with txt files exported for Biotek's Gen5 software with the following output parameters:
        -Contents: Well data for each read plus OD600. No summary or data reduction information.
        -Format: Can inlcude Headings, Matrix column & row labels. Seperator is Tab.
    
    Args:
        -data_file(str): File path to the plate reader data.
        -n_rows(int): Number of plate rows represented in the data.
        -n_columns(int): Number of plate columns represented in the data.
        -total_run_time(float): Total reader run time in hours.
        -sampling_rate(float): Sampling rate in hours.
    Returns:
        -data_dict(dict): Keys are the read titles and values are the 3D data arrays.
        -time_dict(dict): Keys are the read titles and values are 1D arrays of the timepoints in hours.
    '''
    data_read = open(data_file, encoding='iso-8859-1')

    n_time_points = int(total_run_time/sampling_rate)

    data_dict = {}
    time_dict = {}

    for i, line in enumerate(data_read):
        if debug: #debug:
            print(i, line, len(line))
        if  line[0:4] == 'Read' or line[0:3] == 'GFP' or line[0:3] == 'RFP' or line[0:3] == '600' or line[0:5] == 'Ratio' or len(line) == 4 or len(line) == 8:
            time_i = -1
            read = str(line)[:-1]

            data_dict[read] = np.zeros((n_rows, n_columns, n_time_points+1))
            time_dict[read] = np.zeros(n_time_points+1)
            data_dict[read][:] = np.nan
            time_dict[read][:] = np.nan
        
        elif  line != '\n' and 'Time' not in line:
            element = line.strip().split('\t') # Remove \n from lines and convert to strings.
            if debug:
                print('element, length element:', element, len(element))

            time_i += 1
            
            time_split = str(element[0]).split(':') # Covert time to string and seperate into hr, min, sec.

            s = float(time_split[2])
            m = float(time_split[1])+(s/60)
            h = float(time_split[0])+(m/60)
            
            time_dict[read][time_i] = h

                
            for row_i in range(n_rows):
                for column_i in range(n_columns):
                    if debug:
                        print('i,j,element_i:', row_i, column_i, n_columns*row_i+1+1+column_i)
                    if len(element) == 1:
                        data_dict[read][row_i, column_i, time_i] = np.nan
                    if element[n_columns*row_i+1+1+column_i] == 'OVRFLW':
                        data_dict[read][row_i, column_i, time_i] = np.nan
                    else:   
                        data_dict[read][row_i, column_i, time_i] = element[n_columns*row_i+1+1+column_i]
            if debug:
                print(data_dict[read])

    return(data_dict, time_dict)

def blank(group_names:list, groups:list, blanks:list, reads:list)->dict:
    """
    Blanks the data in goups by suptracting the data in blanks for everytime point.

    Args:
        group_names (list): A list of strings characterizing each of the experimental groups.
        groups (list): A list of dictionaries which contain the read data for each experimental group. The keys are the read type and the values are 3 dimentional numpy arrays of the data.
        blanks (list): A list of dictionaries which contain the read data for each blank group. The nth dictionary contains the read type(key) and data(value) which will be applied to the nth experimental group in groups.
        reads (list): A list of strings which designates the reads of interest.

    Returns:
        dict: The keys are a strings designating the group type and the values are dictionaries containing the blanked data for each read and the error.
    """
    blank_data = {}
    for i, name in enumerate(group_names):
        blank_data[name] = {}
        g = groups[i]
        b = blanks[i]
        for read in reads:
            blank_data[name].update({read: g[read].mean(axis=1)-b[read]})
            blank_data[name].update({read+'_err': np.sqrt(np.square(g[read].std(axis=1))/g[read].shape[1]+np.square(b[read].std(axis=0))/b[read].shape[0])})                                  
    return blank_data 

def normalize(group_names:list, groups:list, reads:list, off_set=0.1)->dict:
    """
    Blanks the data by dividing the reads (excluding the OD600 read) by the OD600 data plus an offset (should be the starting OD).

    Args:
        group_names (list): A list of strings characterizing each of the experimental groups.
        groups (list): A list of dictionaries which contain the read data for each experimental group. The keys are the read type and the values are 3 dimentional numpy arrays of the data.
        reads (list): A list of strings which designates the reads of interest.
        off_set (float, optional): The offest in the denomenator of the equation y = read/(off_set+OD600). Defaults to 0.1.

    Returns:
        dict: The keys are a strings designating the group type and the values are dictionaries containing the normalize data for each read as well as the error.
    """
    norm_data = {}
    for i, name in enumerate(group_names):
        norm_data[name] = {}
        g = groups[i]
        if '_err' not in name:
            norm_data[name].update({read: g[read]/(off_set+g['600']) for read in reads if read != '600'})
            norm_data[name].update({'600': g['600']})
            norm_data[name].update({read+'_err': norm_data[name][read]*np.sqrt(np.square(g[read+'_err']/(off_set+g[read]))+np.square(g['600_err']/(off_set+g['600']))) for read in reads if read != '600'})
            norm_data[name]['600_err'] = g['600_err']
    return norm_data

def well_curves(data:dict, time:dict, read:str, size=(20, 15), s=10)->None:
    """
    Generates a figure where each suplots shows the read curve (designated by read) of each well.

    Args:
        data (dict): Keys are the read titles and values are the 3D data arrays.
        time (dict): Keys are the read titles and values are 1D arrays of the timepoints in hours.
        read (str): A list of strings which designates the reads of interest.
        size (tuple, optional): Matlab figure size. Defaults to (20, 15).
        s (int, optional): Scatter plot marker size. Defaults to 10.
    """
    fig, axs = plt.subplots(data[read].shape[0], data[read].shape[1], figsize=size, sharey=True, sharex=True)
    
    for i in range(data[read].shape[0]):
        for j in range(data[read].shape[1]):
            axs[i,j].scatter(time[read],data[read][i,j,:],s=s)
            
    fig.suptitle(read, size=24)
    fig.supxlabel('Time [Hr.]', size=24)
    plt.tight_layout()
    
def pretty_plot(groups:list, time:dict, reads:list, titles:list, err='fill', spines=False,
                a=1, fcorr=0.1, OD600=True, scale=3, e_every=20, csize=5,
                pad=5, fsize=18, face_size=1, ylim=False, read_names=None,
                read_f_offset=1, save=None, colors=None)->None:
    """Create a figure with contains read plots for each experimental group designated by groups. Each column designates a seperate experimental 

    Args:
        groups (list): List of dictionaries for each group where the keys are the read type and the values are a 3D numpy array of the data and errors.
        time (dict): Keys are the read type and values are a numpy array of the the timepoints that correspond to the data in the group dictionaries.
        reads (list): List of the read types.
        titles (list): List of the names of the experimental groups.
        err (str, optional): String describing whether to use error bars or fills to represent the error. Defaults to 'fill'.
        spines (bool, optional): Whether or not to include the top and right spines for each subplot. Defaults to False.
        a (int, optional): Alpha value for the marker and error fills. Defaults to 1.
        fcorr (float, optional): Correction to the error fill alpha value. Defaults to 0.1.
        OD600 (bool, optional): Choose whether or not to include the OD600 curves. Defaults to True.
        scale (int, optional): _description_. Defaults to 3.
        e_every (int, optional): _description_. Defaults to 20.
        csize (int, optional): _description_. Defaults to 5.
        pad (int, optional): _description_. Defaults to 5.
        fsize (int, optional): _description_. Defaults to 18.
        face_size (int, optional): _description_. Defaults to 1.
        ylim (bool, optional): _description_. Defaults to False.
        read_names (_type_, optional): _description_. Defaults to None.
        read_f_offset (int, optional): _description_. Defaults to 1.
        save (_type_, optional): _description_. Defaults to None.
        colors (_type_, optional): _description_. Defaults to None.
    """
    
    reads = copy.deepcopy(reads)
    legend = []
    p = []
    f = []
    read_len = len(reads)
    n_group = len(groups)
    
    fig, axs = plt.subplots(read_len, n_group, figsize=(4*scale,read_len*scale), sharey='row', sharex=True)
    
    if not OD600:
        reads.remove('600')
        
    for i, group in enumerate(groups):
        for j, read in enumerate(reads):
            for k in range(group[read].shape[0]):
                if colors == None:
                    c = None
                else:
                    c = colors[k]
                if err == 'fill':
                    p_element = axs[j,i].scatter(time[read], group[read][k,:], alpha=a, s=face_size, color=c)
                    p.append(p_element)
                    
                    color = p_element.get_facecolor()
                    
                    f_element = axs[j,i].fill_between(time[read], group[read][k,:]-group[read+'_err'][k,:], 
                                                      group[read][k,:]+group[read+'_err'][k,:], 
                                                      color=color,alpha=a*fcorr)
                    f.append(f_element)
                if err == 'bar':
                    axs[j,i].errorbar(time[read], group[read][k,:],
                                        group[read+'_err'][k,:], errorevery=20, capsize=csize)
                    
                if not spines:
                    axs[j,i].spines['right'].set_visible(False)
                    axs[j,i].spines['top'].set_visible(False)
                    
            axs[0,i].set_title(titles[i], fontsize=fsize, loc='left')
            
            if ylim:
                axs[0,i].set_ylim(ylim)

    
    axs[1,n_group-1].legend(zip(p, f), ['10 mM' ,'1 mM', '0.1 mM', '0 mM'], fontsize='medium', frameon=False, title='IPTG Conc.')    
    
    if read_names != None:
        for i, read_name in enumerate(read_names):
            axs[i,0].annotate(read_name, xy=(0, 0.5), xytext=(-axs[0,0].yaxis.labelpad-pad, 0),
                            xycoords=axs[i,0].yaxis.label, textcoords='offset points',
                            size=fsize*read_f_offset, ha='right', va='center', rotation=90)
    else:
        for i, read in enumerate(reads):
            if read == '600':
                read = 'OD$_{600}$'    
            else:
                read = read[:3]
            axs[i,0].annotate(read, xy=(0, 0.5), xytext=(-axs[0,0].yaxis.labelpad-pad, 0),
                            xycoords=axs[i,0].yaxis.label, textcoords='offset points',
                            size=fsize*read_f_offset, ha='right', va='center', rotation=90)   
   
    fig.supxlabel('Time [Hr.]', fontsize=fsize*read_f_offset)#, y=0.04)
    
    if save != None:
        plt.savefig(save, transparent=True, dpi=1200)
    plt.show()