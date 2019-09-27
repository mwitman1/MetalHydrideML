#! /usr/bin/env python

# A module for curating the DOE H2 storage materials database
# Interfaces with ICSD to attempt to extract additional properties
import re
import numpy as np
import pandas as pd
import sys
from magpie import MagpieServer
import inspect
import pickle
import time
import sys
from copy import deepcopy
from pprint import pprint
import requests
from joblib import dump,load

import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge

import pymatgen
from pymatgen.io.cif import CifParser
from pymatgen.core.composition import Composition
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator as VIRE
from pymatgen.core.periodic_table import Element

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
prop_cycle = plt.rcParams['axes.prop_cycle']
DEFAULT_COLORS = prop_cycle.by_key()['color']
plt.rc('text', usetex=True)
plt.rc('font', **{'size':10})

####################################################################################
# A bunch of utility functions
####################################################################################

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def get_random_RF_param_grid():
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    return random_grid

def retain_features(X,feature_list):
    
    return X[feature_list]

def compute_mean_relative_error(y_true, y_pred):

    return np.average(np.abs((y_pred-y_true)/y_true)*100)
    
def draw_y_equals_x(ax):

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    ax.plot([max(xlim[0],ylim[0]), min(xlim[1],ylim[1])],
            [max(xlim[0],ylim[0]), min(xlim[1],ylim[1])],
            linestyle='--',
            c='black',
            linewidth=0.5)

def cluster_feature_vs_feature(x,y,min_samples=3, eps=10): 

    X = np.vstack((x,y)).T
    clust = OPTICS(min_samples=min_samples, xi=0.1,min_cluster_size=3)
    clust.fit(X)
    labels_optics = clust.labels_[clust.ordering_]
    labels_dbscan = cluster_optics_dbscan(reachability=clust.reachability_,
                               core_distances=clust.core_distances_,
                               ordering=clust.ordering_, eps=eps)
    final_labels = labels_dbscan
    print("Computed %d classes"%len(np.unique(final_labels)))
    return X, final_labels

def column_to_label(colname):

    if colname == 'Heat_of_Formation_kJperMolH2':
        return r"$\Delta H$ [kJ/mol H$_2$]"
    elif colname == 'Entropy_of_Formation_kJperMolH2perK':
        return r"$\Delta S$ [J/(mol H$_2$ $\cdot$ K)]"
    elif colname == 'Equilibrium_Pressure_25C':
        return r"$P_{eq}^o$"
    elif colname == 'LnEquilibrium_Pressure_25C':
        return r"$\ln P_{eq}^o (25 ^oC)$"
    elif colname == 'volume':
        return r"$V_{cell}$ [\AA$^3$]"
    elif colname == 'volume_ps':
        return r"$\nu_{pa}^{MP}$ [\AA$^3$/atom]"
    elif colname == 'volume_ps_generic':
        return r"$\nu_{pa}$ [\AA$^3$/atom]"
    elif colname == 'mean_GSvolume_pa':
        return r"$\nu_{pa}^{Magpie}$ [\AA$^3$/atom]"
    elif colname == 'empty_volume_ps':
        return r"$\nu_{pa}-\bar{V}_{atom}$ [\AA$^3$/atom]"
    elif colname == 'mean_CovalentRadius':
        return r"mean$\_$CovalentRadius"
    elif colname == 'mean_SpaceGroupNumber':
        return r"mean$\_$SpaceGroupNumber"
    elif colname == 'energy_per_atom':
        return r"$E_{atom}$"
    elif colname == 'formation_energy_per_atom':
        return r"$E_{f,atom}$"
    elif colname == 'mean_Electronegativity':
        return r"mean$\_$Electronegativity"
    elif colname == 'most_Electronegativity':
        return r"most$\_$Electronegativity"
    elif colname == 'mean_MeltingT':
        return r"mean$\_$MeltingT"
    elif colname == 'normalized_delH':
        return r"$\Delta H / (RT^o)$"
    elif colname == 'normalized_delS':
        return r"$\Delta S / R$"
    else :
        print("Add translation for %s"%colname)
        return colname

def filter_by_predict_value(limlower, limupper, y, holdlower=True, holdupper=True):

    if limlower is not None and limupper is None:
        holdout_indices = np.where(y<limlower)
        keep_indices    = np.where(y>limlower)
    elif limlower is None and limupper is not None:
        holdout_indices = np.where(y>limupper)
        keep_indices    = np.where(y<limupper)
    elif limlower is not None and limupper is not None:
        if holdlower and holdupper:
            holdout_indices = np.where((y<limlower) | (y>limupper))[0]
        elif holdlower and not holdupper:
            holdout_indices = np.where(y<limlower)
        elif not holdlower and holdupper:
            holdout_indices = np.where(y>limupper)
        else:
            holdout_indices = np.array([],dtype=int)

        keep_indices    = np.where((y>limlower) & (y<limupper))[0]
    else:
        keep_indices = slice(0,len(y))
        holdout_indices = np.array([],dtype=int)

    return keep_indices, holdout_indices 

def filter_by_predict_value_vs_value(x, y, limlowerx=None,limupperx=None,
                                     limlowery=None,limuppery=None,
                                     discard_nan = True):

    originalind = np.arange(0,len(x),1)
    
    keep, hold = filter_by_predict_value(limlowerx,limupperx,x)

    x = x[keep]
    y = y[keep]
    originalind=originalind[keep]

    keep, hold = filter_by_predict_value(limlowery,limuppery,y)

    x = x[keep]
    y = y[keep]
    originalind=originalind[keep]

    mask = ~np.isnan(x)

    x = x[mask]
    y = y[mask]
    originalind=originalind[mask]

    mask = ~np.isnan(y)

    x = x[mask]
    y = y[mask]
    originalind=originalind[mask]


    return x, y, originalind


def plot_feature_vs_feature(df1, feature1, df2, feature2, df3 = None, feature3 = None,
                                                          limlower1=None,limupper1=None,
                                                          limlower2=None,limupper2=None,
                                                          cluster=False,
                                                          display_SC=False,
                                                          special_volume_fig=False,
                                                          figsize=(3.3,2.0),
                                                          specialdelHvsdelS=False,
                                                          specialdelHvsdelSv2=False,
                                                          drawyequalsx=False):
    """
    Custom function to plot some of the specific metal hydride screening results

    plot df1['feature1'] vs. df2['feature2'], color coded by df3['feature3']
    """

    if special_volume_fig:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2.8,2.6),sharey=True,
                               gridspec_kw={'width_ratios': [2.25, 1]})
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax = np.array([ax])


    # Get x and y features to plot, and the indices to keep 
    x = np.array(getattr(df1, feature1),dtype=float)
    y = np.array(getattr(df2, feature2),dtype=float)
    x, y, originalind = filter_by_predict_value_vs_value(x, y, limlower1, limupper1, 
                                                         limlower2,limupper2,discard_nan=True)

    # Here provide special options for manipulating various quantities in the dataset
    if specialdelHvsdelS:
        x/=8.314159
        y/=(8.314159*298/1000)

    # set the color coding based on a thrid feature, if provided
    if df3 is not None and feature3 is not None:
        try:
            # the value we want to color code on is a float
            c = np.array(getattr(df3, feature3),dtype=float)
            c = c[originalind]
            continuous_c = True
        except:
            # the value we want to color code on is a string
            allstrs = np.array(getattr(df3, feature3))[originalind]
            uniquestrs = np.unique(allstrs)
            str_to_c = dict([(uniquestrs[i],DEFAULT_COLORS[i]) for i in range(len(uniquestrs))])
            c = np.array([str_to_c[s] for s in allstrs])
            continuous_c = False
            # in this case also need to manually prepare the legend
            legend_elements=[Line2D([0],[0],marker='o',color='none',label=uniquestrs[i],
                                    markerfacecolor=str_to_c[uniquestrs[i]])\
                             for i in range(len(uniquestrs))]
    else:
        c = 'blue'

    # Compute the spearman correlation coefficient
    SC = stats.spearmanr(x,y)

    # Cluster the features if requested
    if cluster:
        X, labels = cluster_feature_vs_feature(x,y)
        numclasses = len(np.unique(labels))
        colors = ['b.', 'r.', 'g.', 'y.', 'c.','m.']
        #colors = [DEFAULT_COLORS[i] for i in range(numclasses-1)] 
        it=0
        for klass, color in sorted(zip(range(0, numclasses), colors),
                                   key=lambda x: len(X[labels == x[0]]),reverse=True):
            Xk = X[labels == klass]
            ax[0].plot(Xk[:, 0], Xk[:, 1], colors[it], alpha=0.3,markersize=10)
            it+=1
        ax[0].plot(X[labels == -1, 0], X[labels == -1, 1], 'k+', alpha=0.3,markersize=10,label="Unassigned")
        #ax.set_title('Automatic Clustering\nOPTICS')
        ax[0].set_xlabel(column_to_label(feature1))
        ax[0].set_ylabel(column_to_label(feature2))

        if special_volume_fig:
            # plot the Cuevas data
            cuevasdata = np.loadtxt("CUEVAS/Cuevas2002_H2PressureVsCellV.csv",
                                    dtype='float',
                                    delimiter=',')
            ax[0].scatter(cuevasdata[:,0],np.log(cuevasdata[:,1]*0.986923), marker="*", 
                       s=100, edgecolors='black',facecolors='none',linewidth=1, label="Cuevas")

            smithdata = np.loadtxt("SMITH/Smith1983_H2PressureVsVsite.csv",dtype='float',delimiter=',')
            ax[0].scatter(smithdata[:,0]*29,smithdata[:,1], marker="s", 
                       s=70, edgecolors='green',facecolors='none',linewidth=1.5, label="Smith")

            # divide volume by number of sites in the lattice
            volume_ps = np.array(getattr(df1, "volume_ps"),dtype=float)[originalind]
            nsites = np.array(getattr(df1, "nsites"),dtype=float)[originalind]

            # plot the same data (and classes) as before but volume normalized by nsites           
            X[:,0] = X[:,0]/nsites 
            it = 0
            for klass, color in sorted(zip(range(0, numclasses), colors),
                                       key=lambda x: len(X[labels == x[0]]),reverse=True):
                Xk = X[labels == klass]
                ax[1].plot(Xk[:, 0], Xk[:, 1], colors[it], alpha=0.3,markersize=10)
                it+=1
            ax[1].plot(X[labels == -1, 0], X[labels == -1, 1], 'k+', alpha=0.3,markersize=10)
            ax[1].scatter(cuevasdata[:,0]/6,np.log(cuevasdata[:,1]*0.986923), marker="*", 
                       s=100, edgecolors='black',facecolors='none',linewidth=1, label="Cuevas")
            ax[1].scatter(smithdata[:,0],smithdata[:,1], marker="s", 
                       s=70, edgecolors='green',facecolors='none',linewidth=1.5, label="Smith")

            ax[1].set_xlabel(column_to_label("volume_ps"))
            #ax[1].set_ylim((ax[1].get_ylim()[0]+0.2,None))

        #ax[0].set_ylim((ax[0].get_ylim()[0]+0.2,None))
        ax[0].legend(loc='best',prop={'size':8},borderpad=0.15,handletextpad=0.25)
    
    # If no clustering requested, simply plot y equals x
    else:
        if isinstance(c,str):

            if display_SC:
                ax[0].scatter(x, y, edgecolor='blue', linewidths=1, alpha=0.3, 
                              label="SC = %.2f"%(SC[0]))
                ax[0].legend(loc='best', prop={'size':8},borderpad=0.15,handletextpad=0.25)
            else:
                ax[0].scatter(x, y, edgecolor='blue', linewidths=1, alpha=0.3)
        else:
            if continuous_c:
                vals = ax[0].scatter(x, y, c=c, cmap='viridis',linewidths=1, alpha=0.8, 
                                     label="SC = %.2f"%(SC[0]),edgecolor='black',linewidth=0.3)
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes('right', size='7%', pad=0.05)
                cbar = fig.colorbar(vals, cax=cax,orientation='vertical')
                cbar.set_label(column_to_label(feature3))
                if display_SC:
                    ax[0].legend(loc='best', prop={'size':8},borderpad=0.15,handletextpad=0.25)
            else:
                vals = ax[0].scatter(x, y, c=c, linewidths=1, alpha=0.8, 
                                     label="SC = %.2f"%(SC[0]),edgecolor='black',linewidth=0.3)
                ax[0].legend(bbox_to_anchor=(1.05,1),handles=legend_elements,loc='upper left',
                             borderaxespad=0.)

        # special option to show delH vs delS with contours of constant lnPeqo
        if specialdelHvsdelSv2:
            lnPeqos = [-15,-5,5]
            delSrange = np.linspace(0,200)
            delH = [(-8.314159*P+delSrange)*298.15/1000 for P in lnPeqos]

            for P, series in zip(lnPeqos, delH):
                ax[0].plot(delSrange,series,label=r"$\ln P_{eq}^o = %.0f$"%P)
            
            ax[0].legend(loc="best",handlelength=0.75,handletextpad=0.5)
        
        ax[0].set_xlabel(column_to_label(feature1))
        ax[0].set_ylabel(column_to_label(feature2))
        ax[0].set_xlim((limlower1,limupper1))
        ax[0].set_ylim((limlower2,limupper2))


        # this option plots the correlation of both enthalpy and entropy with lnPeqo
        # with some special formatting required to visualize the data properly
        if(specialdelHvsdelS):
            # reset figure
            plt.close()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            z = np.array(getattr(df2, "LnEquilibrium_Pressure_25C"),dtype=float)[originalind]

            # concatenate delH and delS and randomly scramble the order
            allzs = np.array(list(z)+list(z))
            allvals = np.array(list(y)+list(x))
            allcolors = np.array(['red' for _ in range(len(x))]+['green' for _ in range(len(x))])
            p = np.random.permutation(len(allcolors))

            # Actual data
            ax.scatter(allzs[p], allvals[p], color=allcolors[p],
                       edgecolor='black',linewidth=0.3,s=10)
            # Dummy data for the legend
            ax.scatter([],[],color='red',edgecolor='black',
                       linewidth=0.3,label=column_to_label('normalized_delH'))
            ax.scatter([],[],color='green',edgecolor='black',
                       linewidth=0.3,label=column_to_label('normalized_delS'))

            ax.set_xlabel(column_to_label("LnEquilibrium_Pressure_25C"))
            ax.set_ylabel(r'$Dimensionless$')
            maxy = max(max(y),max(x))
            ax.set_ylim((None,maxy+0.1*maxy))
            ax.legend(loc="best",borderpad=0.15,handletextpad=0.25)


        # Draws the parity line for variables that should be y=x
        if drawyequalsx:
            draw_y_equals_x(ax[0])

    plt.tight_layout(pad=0.3)
    plt.show()

    #print out the compositions and classes of these materials
    #try:
    #    print("Compositions plotted (%d):"%len(np.array(df2["Composition_Formula"])[originalind]))
    #    print(np.array(df2["Composition_Formula"])[originalind])
    #    print("Classes plotted:")
    #    print(np.array(df2["Material_Class"])[originalind])
    #except:
    #    pass
   


def plot_prediction_vs_custom_volumes(df1, df2, limlower1=None,limupper1=None,
                                                 limlower2=None,limupper2=None,
                                                 cluster=False,
                                                 display_SC=False):

    x = np.array(getattr(df1, "mean_GSvolume_pa"),dtype=float)
    y = np.array(getattr(df2, "LnEquilibrium_Pressure_25C"),dtype=float)
    x, y, originalind = filter_by_predict_value_vs_value(x, y, limlower1, limupper1, 
                                                         limlower2,limupper2,discard_nan=True)
    SC = stats.spearmanr(x,y)

    x1 = np.array(getattr(df1, "volume_ps"),dtype=float)
    y1 = np.array(getattr(df2, "LnEquilibrium_Pressure_25C"),dtype=float)
    x1, y1, originalind1 = filter_by_predict_value_vs_value(x1, y1, limlower1, limupper1, 
                                                         limlower2,limupper2,discard_nan=True)
    SC1 = stats.spearmanr(x1,y1)

    cuevasdata = np.loadtxt("CUEVAS/Cuevas2002_H2PressureVsCellV.csv",dtype='float',delimiter=',')
    SC_cuevas = stats.spearmanr(cuevasdata[:,0],cuevasdata[:,1])

    smithdata = np.loadtxt("SMITH/Smith1983_H2PressureVsVsite.csv",dtype='float',delimiter=',')
    SC_smith = stats.spearmanr(smithdata[:,0],smithdata[:,1])

    figsize=(3.3,2.6)
    figsize=(2.3,2.6)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)

    if display_SC:
        label = r"Magpie (%.2f)"%(SC[0])
        label1 = r"MP (%.2f)"%(SC1[0])
        label2 = r"Cuevas (%.2f)"%(SC_cuevas[0])
        label3 = r"Smith (%.2f)"%(SC_smith[0])
    else:
        label = r"Magpie"
        label1 = r"MP"
        label2 = r"Cuevas"
        label3 = r"Smith"

    ax.scatter(x, y, edgecolor='blue', linewidths=1, alpha=1, label=label)
    ax.scatter(x1, y1, edgecolor='red', linewidths=1, alpha=1, label=label1)
    ax.scatter(cuevasdata[:,0]/6,np.log(cuevasdata[:,1]*0.986923), marker="*", 
               s=100, edgecolors='black',facecolors='none',linewidth=1.0, label=label2)
    ax.scatter(smithdata[:,0],smithdata[:,1], marker="s", 
               s=70, edgecolors='green',facecolors='none',linewidth=1.5, label=label3)

    
    ax.legend(loc='best', prop={'size':8},borderpad=0.15,handletextpad=0.25)
   
    ax.set_xlabel(column_to_label("volume_ps_generic")) 
    ax.set_ylabel(column_to_label("LnEquilibrium_Pressure_25C")) 
    #ax.set_ylim((ax.get_ylim()[0]+0.8,None))

    plt.tight_layout(pad=0.3)
    plt.show()

def plot_delHprediction_vs_custom_volumes(df1, df2, limlower1=None,limupper1=None,
                                                 limlower2=None,limupper2=None,
                                                 cluster=False,
                                                 display_SC=False):

    x = np.array(getattr(df1, "mean_GSvolume_pa"),dtype=float)
    y = np.array(getattr(df2, "Heat_of_Formation_kJperMolH2"),dtype=float)
    x, y, originalind = filter_by_predict_value_vs_value(x, y, limlower1, limupper1, 
                                                         limlower2,limupper2,discard_nan=True)
    SC = stats.spearmanr(x,y)

    x1 = np.array(getattr(df1, "volume_ps"),dtype=float)
    y1 = np.array(getattr(df2, "Heat_of_Formation_kJperMolH2"),dtype=float)
    x1, y1, originalind1 = filter_by_predict_value_vs_value(x1, y1, limlower1, limupper1, 
                                                         limlower2,limupper2,discard_nan=True)
    SC1 = stats.spearmanr(x1,y1)

    figsize=(3.3,2.6)
    figsize=(2.3,2.6)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=figsize)

    if display_SC:
        label = r"Magpie (SC = %.2f)"%(SC[0])
        label1 = r"MP (SC = %.2f)"%(SC1[0])
    else:
        label = r"Magpie"
        label1 = r"MP"
        
    ax.scatter(x, y, edgecolor='blue', linewidths=1, alpha=1, label=label)
    ax.scatter(x1, y1, edgecolor='red', linewidths=1, alpha=1, label=label1)
    ax.legend(loc='best', prop={'size':8},borderpad=0.15,handletextpad=0.25)
   
    ax.set_xlabel(column_to_label("volume_ps_generic")) 
    ax.set_ylabel(column_to_label("Heat_of_Formation_kJperMolH2")) 
    #ax.set_ylim((ax.get_ylim()[0]+0.8,None))

    plt.tight_layout(pad=0.3)
    plt.show()

def bargraph_on_class(df):
    all_classes = df["Material_Class"].value_counts()
    print(all_classes.index)
    print(list(all_classes))

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(3.3,2.6))
    ax.bar([i for i in range(1,len(all_classes)+1)],
           height=all_classes,
           tick_label = all_classes.index)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.set_ylabel("Number of Entries")

    plt.tight_layout(pad=0.3)
    plt.show()

    


####################################################################################
# Main class for storing/doing ML on metal hydride data


class H2Data(object):
    
    def __init__(self, fname, remove_nan_attr, remove_classes, headerlength=6):

        self.R = 0.008314159 # kJ / (mol K)       
        self.fname           = fname
        self.remove_nan_attr = remove_nan_attr
        self.remove_classes  = remove_classes
        self.headerlength    = headerlength

        self.MP_API_key = 'WffzTTad1Lyx6dHEVRfI'



    def prepare_database_and_features(self, save=True, dbname="curr_database.pkl",
                                      featuresname = "curr_magpie_features.pkl"):

        self._database = pd.read_csv(self.fname,header=self.headerlength)
 
        # 1. Data removal
        print("Num entries: %d" % len(self._database))        

        self.clean_nans(self.remove_nan_attr)
        print("Num entries: %d" % len(self._database))        

        self.clean_classes(self.remove_classes)
        print("Num entries: %d" % len(self._database))

        self.clean_Mm(action='delete')
        print("Num entries: %d" % len(self._database))

        self.clean_wt_percent_compositions(action='delete')
        print("Num entries: %d" % len(self._database))

        self.clean_composition_formula()
        print("Num entries: %d" % len(self._database))

        # 2. Data supplementation 

        # compute entropy of formation and backcalculate the equilibrium pressure at 25C
        self.compute_entropy_of_formation()

        # Check for how well the various properties agree between identical entries
        #self.analyze_duplicates(verbose=False)


        # 3. Magpie features/properties
        self._features = self.extract_magpie_attributes_live(list(self._database['Composition_Formula']),
                                                             method='oqmd-dH')

        if save:
            pickle.dump(self._features, open(featuresname,"wb"))
            pickle.dump(self._database, open(dbname,"wb"))


    def prepare_database_and_features_HS(self, filename, save=True, dbname="HS_database.pkl",
                                         featuresname = "HS_magpie_features.pkl"):
        self._database_HS = pd.read_csv(filename, header=self.headerlength)

        self._features_HS = self.extract_magpie_attributes_live(list(self._database_HS['Composition_Formula']),
                                                                method='oqmd-dH')
                                                                
        if save:
            pickle.dump(self._features_HS, open(featuresname,"wb"))
            pickle.dump(self._database_HS, open(dbname,"wb"))
   

    def prepare_MP_features(self, database, save=True, fname="MP_features.pkl"):

        MP_features = {}

        for formula in database['Composition_Formula']:
        #for formula in ["Th2Al"]:
            #print(formula)
            try:
                r = requests.get("https://www.materialsproject.org/rest/v2/materials/%s/vasp?API_KEY=nOVGnRHX7T6rjsi6"%formula,verify=False)
                MP_features[formula]=r.json()
            except:
                MP_features[formula]="Error"

        if save:
            pickle.dump(MP_features, open(fname,"wb"))

        return MP_features
                                                             
    ########################################################################################
    # Functions to clean HYDPARK database
    # For now discard materials for which significant extra work would have to be done
    # to make them amenable for the ML workflow
    
    def clean_nans(self, attr):
        """
        Remove a row from the database if any entry in attr is nan
        """
        print("Removing rows with nan's for any of the following attributes: %s"%str(attr))

        for at in attr:
            #print(at, self._database[at])
            self._database = self._database[pd.to_numeric(self._database[at], errors='coerce').notnull()]
            # TODO should figure out how to do it this way
            #self._database = self._database[getattr(self._database,at).apply(lambda x: x.isnumeric())]


    def clean_substr_from_column(self, colref, substr):
        self._database = self._database[~colref.str.contains(substr)]


    def clean_classes(self, classes):
        """Remove materials of a specific class type"""
        print("Removing rows with class %s"%str(classes))
        for c in classes:
            # TODO ?
            pass 


    def clean_Mm(self, action):
        """
        Some Hydpark entries have Composition Formulas containing Mischmetal (Mm) or 
        Lanthanum rich Mischmetal (Lm) for which descriptor generation won't work right out of the box
        (https://en.wikipedia.org/wiki/Mischmetal)

        """

        print("Action to take for any Composition Formula with Mm: %s"%str(action))

        if action == 'delete':
            self.clean_substr_from_column(self._database["Composition_Formula"],"Mm")
            self.clean_substr_from_column(self._database["Composition_Formula"],"Lm")
        elif action == 'replace':
            # TODO We can account for 98% of Mischmetal composition (55% cerium, 25% lanthanum, 18% neodymium)
            # so we might be able to replace Mm for (Ce0.55La0.25Nd0.18)??
            pass

    def clean_wt_percent_compositions(self,action):
        """
        Some Hydpark entries give composition by weight percent, indicated by a -
        e.g. Mg-10Ni
        There's only a few of them so let's delete them for now but they could 
        converted to stoichiometry
        """
    
        print("Action to take for any Composition Formula in wt percent (contains -): %s"%str(action))

        if action == 'delete':
            self.clean_substr_from_column(self._database["Composition_Formula"],"-")
        elif action == 'repalce':
            pass


    def clean_composition_formula(self):

        # store a copy of the original compositions
        self._database["Original_Composition_Formula"] = self._database["Composition_Formula"]

        # use pymatgen to get get the reduced composition
        setattr(self._database, 'Composition_Formula', 
                getattr(self._database,'Composition_Formula').apply(self.pymatgen_reduce_composition))

        # Pymatgen seems to process M as a valid element by converting to M0+
        # TODO what does this mean?? For now get rid of it
        self.clean_substr_from_column(self._database["Composition_Formula"],"-")
        self.clean_substr_from_column(self._database["Composition_Formula"],"\+")

        # TODO, figure out what this entry (what is M1.006 ??) is and how to fix it...
        self.clean_substr_from_column(self._database["Composition_Formula"],"M1.")

        # to make complex hydride compositions consistent with the rest of the database, need to strip
        # the hydrogens bc the hydrogenated stoichiometries only reported for complex
        setattr(self._database, 'Composition_Formula', 
                getattr(self._database,'Composition_Formula').apply(self.clean_Hs_from_complex))

        # pymatgen can also do this but I already wrote it so why not
        #print("Start formula: " + formula)
        #allcompositions = re.findall(r"[-+]?\d*\.\d+|\d+", splitformula[0]) 
        #allelements = []

        #prevind = 0
        #for comp in allcompositions: 
        #    # index returns the first find
        #    ind = splitformula[0][prevind:].index(comp)+prevind
        #    #print(prevind, ind)
        #    allelements.append(splitformula[0][prevind:ind])
        #    prevind=ind+1

        ##print(allcompositions)
        ##print(allelements)
        ##print(splitformula[0])
        #return splitformula[0]

    def clean_Hs_from_complex(self,formula):
        formula = formula.split()[0]
        c = Composition(formula)
        dict_rep = c.to_reduced_dict
        keys = dict_rep.keys()
        if 'H' in keys:
            # since Composition doesn't support item assignment/deletion
            newdict = {}
            for key in keys-'H':
                newdict[key] = dict_rep[key]
            newdict = ["%s%.4f"%tup for tup in newdict.items()]
            returnstr = "".join(newdict)
            print("Cleaning %s to %s"%(formula, returnstr))
        else:
             returnstr = formula
        
        return returnstr


    def pymatgen_reduce_composition(self, formula):
        # TODO what does the "formula (M)" mean? for now discard (M)
        # use something like pymatgen Composition module or the following custom implementation 
        # http://folk.ntnu.no/haugwarb/TKP4106/Tore/Syllabus/topic_03/index.html

        formula = formula.split()[0]
        c = Composition(formula)
        returnstr = ""
        if 'M0+' in c.to_reduced_dict.keys():
            # TODO, figure out what this entry (what is M1.006 ??) is and how to fix it...
            # Why is key stored as M0+ but then the reduced formula converts it back to M...
            returnstr = "".join(["".join([str(i) for i in elem]) for elem in c.to_reduced_dict.items()])
        else:
            returnstr = c.reduced_formula    
        #print(returnstr)
        return returnstr


    def check_composition_formula_cleaning(self, original, cleaned):
        pass 


    def analyze_duplicates(self, verbose=True):
        
        duplicates = self._database[self._database.duplicated(subset='Composition_Formula')]
        unique_compositions = set(duplicates['Composition_Formula'])
        if verbose:
            print("Total entries: %d"%(len(self._database)))
            print("Num duplicate entries found: %d"%(len(duplicates)+len(unique_compositions)))
            print("Of which are unique: %d"%len(unique_compositions))
        #print(duplicates[['Composition_Formula','Temperature_oC','Equilibrium_Pressure_25C','Pressure_Atmospheres_Absolute']])

        # Check consistency of data from duplicate entries that SHOULD be independent 
        # from who was performing the experiment ...
        plot_enthalpy = []
        plot_entropy  = []
        plot_eqlbPress= []

        for comp in unique_compositions:
            
            dup_data = self._database.loc[self._database['Composition_Formula'] == comp]
            enthalpy = np.array(dup_data['Heat_of_Formation_kJperMolH2'],dtype=float)
            entropy  = np.array(dup_data['Entropy_of_Formation_kJperMolH2perK'],dtype=float)
            eqlbPress= np.array(dup_data['Equilibrium_Pressure_25C'],dtype=float)


            if verbose:
                print("Duplicate composition (%d entries): %s"%(len(dup_data),comp))
                print("HofF = %.2f +/- %.2f, EofF = %.4f +/- %.4f, EqPress25C = %.1e +/- %.1e"%\
                      (np.average(enthalpy), np.std(enthalpy), np.average(entropy), np.std(entropy),
                       np.average(eqlbPress), np.std(eqlbPress)))

            if np.std(enthalpy) != 0.0:
                plot_enthalpy.append(np.average(enthalpy)/  (np.std(enthalpy)))
            if np.std(entropy) != 0.0:
                plot_entropy.append(np.average(entropy)/    (np.std(entropy)))
            if np.std(eqlbPress) != 0.0:
                plot_eqlbPress.append(np.average(eqlbPress)/(np.std(eqlbPress)))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.3,2.5))
        #ax.scatter([1 for _ in plot_enthalpy], plot_enthalpy, c='red')
        #ax.scatter([2 for _ in plot_entropy], plot_entropy, c='green')
        #ax.scatter([3 for _ in plot_eqlbPress], plot_eqlbPress, c='blue')
        ax.boxplot([plot_enthalpy, plot_entropy, plot_eqlbPress])

        ax.set_ylabel(r"$\left< X \right> / \sigma_X ^2$")
        ax.set_yscale('log')
        ax.set_ylim((None,10**4))
        #ax.xaxis.set_ticklabels([column_to_label('Heat_of_Formation_kJperMolH2'),
        #                         column_to_label('Entropy_of_Formation_kJperMolH2perK'),
        #                         column_to_label('LnEquilibrium_Pressure_25C')])
        ax.xaxis.set_ticklabels([column_to_label(r"$X = \Delta H_f$"),
                                 column_to_label(r"$X = \Delta S_f$"),
                                 column_to_label(r"$X = P_{eq}^o$")])
      
        if verbose: 
            print(plot_eqlbPress)
            print(min(plot_eqlbPress))
        plt.tight_layout(pad=0.5) 
        plt.show()



    def clean_duplicates(self, method='median', verbose=True):

        duplicates = self._database[self._database.duplicated(subset='Composition_Formula',keep=False)]
        if verbose:
            print(duplicates["Composition_Formula"])
        unique_compositions = set(duplicates['Composition_Formula'])

        # since features and predict values stored differently, need to reindex/realign so that 
        # we manually remove duplicates consisently between the two dataframes
        newindices = [i for i in range(len(self._database))]
        #self._database.reindex(index=[i for i in range(len(self._database))])
        #self._features.reindex(index=[i for i in range(len(self._features))])
        self._database.index = newindices
        self._features.index = newindices

        if verbose:
            print("%d dupes found with %d unique = %d to remove"%(len(duplicates),len(unique_compositions),
                                                                  len(duplicates)-len(unique_compositions)))

        if method=='first':
            print("Do we really want to do just take the first entry?")
            sys.exit()
        elif method == 'median':
            # by taking the median entry we assume that values on the wing of the distribution
            # were performed poorly and should be discarded

            # if there are only two duplicates we probably have no choice but to take the first?
            final_delete = []
            final_dups = []
            for composition in unique_compositions:
                dup_indices = self._database.index[self._database['Composition_Formula'] == composition].tolist()
                final_dups += dup_indices
            
                #if(len(dup_indices)==2):
                #    final_delete += [dup_indices[0]]
                #    if verbose:
                #        print(composition, dup_indices, dup_indices[0]) 
                #else:
                allheats = np.array(self._database['Heat_of_Formation_kJperMolH2'],dtype=float)[dup_indices]
                medianind_of_dupes = np.argsort(allheats)[len(allheats)//2]
                tokeepind = dup_indices[medianind_of_dupes]
                toremoveind = [ind for ind in dup_indices if ind != tokeepind]
                final_delete += toremoveind
            
                if verbose:
                    print(composition, dup_indices, allheats, medianind_of_dupes, tokeepind, toremoveind)

            self._database = self._database.drop(final_delete) 
            self._features = self._features.drop(final_delete) 

    def clean_specific_composition(self, composition):
        """
        Cleans a composition from both the database and features dataframes
        """

        del_ind = self._database.index[self._database['Composition_Formula'] == composition]

        self._database = self._database.drop(del_ind)
        self._features = self._features.drop(del_ind)
        print(len(self._database))
        print("Deleting composition, index: ", composition, del_ind)

    def clean_heat_of_formation(self, string, method = 'average'):

        string.split('-')
        allreals = re.findall(r"[-+]?\d*\.\d+|\d+", string) 
        np.mean(allreals)
        pass

    ########################################################################################


    
    ########################################################################################
    # Extract/compute relevant MP features 

    def clean_MP_features(self, MP_features, duplicates = "min_energy_per_atom"):

        #success=0
        #for formula in database._MP_features.keys():
        #    if database._MP_features[formula] != "Error" and database._MP_features[formula]["response"]:
        #        pprint(database._MP_features[formula])
        #        success+=1
        #    elif database._MP_features[formula] != "Error" and not database._MP_features[formula]["response"]:
        #        print(formula)
        #print(success)
        #sys.exit()

        print("Parsing MP features and removing duplicates by choosing the lowest energy/atom entry")
        print("Note that some errors were found in MP with the 'nsites' metadata, so I've hard encoded those corrections until further notice")
        tmp = {}
        for formula in MP_features.keys():
            #print(formula)
            
            if MP_features[formula] != "Error":
                data = MP_features[formula]['response']
                #pprint(data)
                # Need to select which entry duplicate to take data from
                # For now we will just choose the lowest energy per atom structure
                energies = [data[i]["energy_per_atom"] for i in range(len(data))]
                volumes = [data[i]["volume"] for i in range(len(data))]
                sites = [data[i]["nsites"] for i in range(len(data))]
                volume_ps = np.array(volumes)/np.array(sites)

                if len(data) > 1:
                    # get all atom energies
                    if duplicates == "min_energy_per_atom":
                        selectind = energies.index(min(energies))
                        #print(volume_ps, volume_ps[selectind])
                    else:
                        print("No method defined for dealing with duplicates")
                elif len(data) == 0:
                    continue
                else:
                    selectind = 0

                # We have to manually overwrite the data for entries we identify as incorrect on MP
                if formula == "MgNi":
                    data[selectind]["volume"]=56.41
                    data[selectind]["nsites"]=4
                    volume_ps[selectind] = 56.41/4
                elif formula == "CeMg2":
                    # TODO: seems like the API requested data doesn't even match the online
                    data[selectind]["volume"]=311.218
                    data[selectind]["nsites"]=12
                    volume_ps[selectind] = 311.218/12
                elif formula == "Mg2Fe":
                    data[selectind]["volume"]=215.022
                    data[selectind]["nsites"]=12
                    volume_ps[selectind] = 215.022/12

                # print the outliers to the trend
                #if(volume_ps[selectind]<12.5 or volume_ps[selectind]>16.75):
                #    print(formula, selectind, sites, sites[selectind], volume_ps, volume_ps[selectind], energies, energies[selectind])    
                print(formula, 
                      np.array(self._database.loc[self._database["Composition_Formula"] == formula,"Material_Class"]), 
                      #np.array(self._features.loc[self._database["Composition_Formula"] == formula,"mean_CovalentRadius"]), 
                      np.array(self._database.loc[self._database["Composition_Formula"] == formula,'LnEquilibrium_Pressure_25C']), 
                      np.array(self._database.loc[self._database["Composition_Formula"] == formula,'Heat_of_Formation_kJperMolH2']), 
                      np.array(self._features.loc[self._database["Composition_Formula"] == formula,'mean_GSvolume_pa']), 
                      volume_ps[selectind],
                      energies[selectind],
)
                      #volume_ps)    


                tmp[formula] = data[selectind] 

        return  tmp 

    def compute_MP_volume_ps(self, MP_features):
        for formula in MP_features.keys():
            data = MP_features[formula]
            volume_ps = np.nan
            try:
                volume_ps = data['volume']/data['nsites']
                mean_CovRad = float(self._features.loc[self._database["Composition_Formula"] == formula,"mean_CovalentRadius"])/100 # In Angstrom
                mean_atom_volume = np.pi*4/3*mean_CovRad**3
                empty_volume_ps = mean_atom_volume/volume_ps
            except:
                pass

            MP_features[formula]["volume_ps"] = volume_ps    
            MP_features[formula]["empty_volume_ps"] = empty_volume_ps    

        return MP_features
            
    
    def compute_MP_volume_descr_1(self, MP_features, method='avgionic',verbose=True):
        # compute the excess volume based on atomic radii (vire.radii)
        # compute the excess volume based on elemental per atom volumes
        possible_methods = ['vire','avgionic']
        if method not in possible_methods:
            raise ValueError("Method %s not yet defined"%method)

        for formula in MP_features.keys():
            data = MP_features[formula]
            
            # somewhat ridiculuous to not have a method to generate from string
            # TODO must be missing it somewhere in documentation
            with open("tmp.cif","w") as f:
                f.write(data["cif"])
            parser = CifParser("tmp.cif")
            struct = parser.get_structures()[0]

            # get the valence ionic radius evaluator
            vire = VIRE(struct)

            # compute "excess volume"
            ucform      = data["unit_cell_formula"]
            cell_vol    = data["volume"]
            ex_vol      = np.nan
            ex_vol_frac = np.nan
            try:
                occ_vol = 0
                for key in ucform.keys():
                    if method == 'vire':
                        radius=vire.radii[key]
                    elif method =="avgionic":
                        radius = Element(key).average_ionic_radius
                    else:
                        sys.exit()
                    occ_vol += ucform[key]*4/3*np.pi*radius**3
                ex_vol = cell_vol - occ_vol
                ex_vol_frac = ex_vol/cell_vol
            except:
                # Warning, if valence detected, pymatgen will include that +2 etc
                # info in the vire.radii keys, but that information is not carried
                # in "unit_cell_formula" keys
                # so for now ignore cases that have valen
                print("%s ignored because of the included valence calculation"%formula)


            MP_features[formula]["ex_vol"]      = ex_vol
            MP_features[formula]["ex_vol_frac"] = ex_vol_frac
            if verbose:
                print(formula, vire.radii, vire.valences, ex_vol_frac)

            return MP_features


    def attach_MP_features_to_all(self, MP_features, feature_list):

        self._features["Composition_Formula"] = self._database["Composition_Formula"]

        for f in feature_list:
            self._database[f] = np.nan

        #print(self._features)
        for formula in self._features["Composition_Formula"]:
            if formula in MP_features.keys():
                #print(MP_features[formula])
                for f in feature_list:
                    self._features.loc[self._features["Composition_Formula"]==formula, f] = MP_features[formula][f]

        self._features = self._features.drop(columns=["Composition_Formula"])
        #print(self._features)
        pass


    ########################################################################################

    
    ########################################################################################
    # Compute additional thermodynamic info based on that given in HYDPARK

    def compute_entropy_of_formation(self):
        """
        Based on database's reported Formation Enthalpy, T, and Eqlb. Pressure,
        compute the entropy of formtion. Then use these to standardize the 
        equilibrium pressure @ 25 C
        """   
        pressure = np.array(self._database['Pressure_Atmospheres_Absolute'],dtype=float)
        enthalpy = np.array(self._database['Heat_of_Formation_kJperMolH2'],dtype=float)        
        temp     = 273.15+np.array(self._database['Temperature_oC'],dtype=float)

        self._database['Entropy_of_Formation_kJperMolH2perK'] =\
            self.R*np.log(pressure)+enthalpy/temp

        self._database['Equilibrium_Pressure_25C'] =\
            np.exp(1/self.R*(self._database['Entropy_of_Formation_kJperMolH2perK']-enthalpy/(298.15)))

    def convert_eqlb_pressure_to_ln(self):
        self._database['LnEquilibrium_Pressure_25C']=np.log(self._database['Equilibrium_Pressure_25C'])

    def manual_modify_thermo(self,composition,delH,delS):
        """
        If we need to modify the thermodynamics of a given composition manually
        E.g. inconsistent phase decomposition was used for reporting thermodynamics of
        Er3Fe23, Ho3Fe23, etc.
        """
        f = 'Heat_of_Formation_kJperMolH2'
        self._database.loc[self._database["Composition_Formula"]==composition,f] = delH
        f = 'Entropy_of_Formation_kJperMolH2perK'
        self._database.loc[self._database["Composition_Formula"]==composition,f] = delS
        f = 'LnEquilibrium_Pressure_25C'
        lnPeq = 1/self.R*(delS-delH/(298.15))
        self._database.loc[self._database["Composition_Formula"]==composition,f] = lnPeq
        f = 'Equilibrium_Pressure_25C'
        Peq = np.nan # Note that we are overriding here, so whatever T, Peq was orginally listed in HYPARK was not what we wanted
        self._database.loc[self._database["Composition_Formula"]==composition,f] = Peq



    ########################################################################################

            

    ########################################################################################
    # Functions to extract features from external sources (i.e. Magpie, Materials Project, OQMD, etc. 

    def extract_magpie_attributes_live(self, formulas, method = 'oqmd-dH', batch_size = 200, 
                                       save=True, dbname = "database.pkl", fname = "features.pkl"):
        """
        formulas   : a list of Composition Formula's for generating features
        method     : magpie supports different ML models (need to investigate if features diff for diff models)
        batch_size : don't flood the server with too big of a data set 
        save       : save the features by hash of formula to avoid querying the server all the time
        """
   
        #formulas = ['(V.9Ti.1).95Fe.05','Fe'] # TODO, check that magpie interprets this correctly
        batch_size = 200
        batches = [formulas[i * batch_size:(i + 1) * batch_size]\
                   for i in range((len(formulas) + batch_size - 1) // batch_size )]
    

        batch_features = []
        for batch in batches:
            print("New %d entry batch:"%len(batch))
            print(batch)
            m = MagpieServer()
            magpie_attr  = m.generate_attributes(method,batch)
            model_result = m.run_model(method,batch)
            # m.generate_attributes and m.run_model return dataframes
            #print(dir(m))
            #print(inspect.signature(m.generate_attributes))
            #print(m.generate_attributes(method,formula))

            #print(type(result))
            #print(dir(result))
            #print(result.columns)
            #print(result.get_values())

            print(magpie_attr)
            print(model_result)
    
            batch_features.append(magpie_attr)
            #for entry in batch:
            #    magpie_features[entry] = {}
            #    magpie_features[entry]['features'] = magpie_attr
            #    magpie_features[entry]['oqmd-dH'] = model_result

            time.sleep(2)

        magpie_features = pd.concat(batch_features)

        return magpie_features

    ########################################################################################
   

     
    ########################################################################################
    # Loading database/features so we don't have to keep querying the servers

    def load_database_and_features(self, dbname, fname):
        """
        Obtain magpie feautures from a locally saved object, useful
        so I'm not constantly querying the magpie server while building this
        """

        self._database = pickle.load(open(dbname,"rb"))
        self._features = pickle.load(open(fname,"rb"))


    def load_database_and_features_HS(self, dbname, fname):
        """
        Obtain magpie feautures from a locally saved object, useful
        so I'm not constantly querying the magpie server while building this
        """

        self._database_HS = pickle.load(open(dbname,"rb"))
        self._features_HS = pickle.load(open(fname,"rb"))

    def load_MP_features(self, fname):
        MP_features = pickle.load(open(fname,"rb"))
        return MP_features

    ########################################################################################


    ########################################################################################
    # Some plotting of database statistics

    def plot_prediction_distribution(self, targets):
        """
        Visualization of whether we have data imbalance
        Plenty of resources for addressing imbalance in classification tasks, but not much on regression
        
        https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28
        """
       
        figsize = (2*len(targets),1.9)
        #figsize = (3.3,1.9) 
        fig, ax = plt.subplots(nrows=1,ncols=len(targets), figsize=figsize,sharey=True)

        keep, exclude = filter_by_predict_value(0, 150, np.array(self._database['Heat_of_Formation_kJperMolH2'],float))
         
        for i in range(len(targets)):
            dist = np.array(self._database[targets[i]],dtype=float)[keep]

            ax[i].hist(dist,alpha=0.8,bins=30)
            ax[i].set_xlabel(column_to_label(targets[i]))
            #ax[i].set_yscale('log')

        ax[0].set_ylabel("Frequency")
        plt.tight_layout(pad=0.5)
        plt.show()

        #fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(3.3,2.6),sharey=True)
        #
        #for i in range(len(targets)):
        #    dist = np.array(self._database[targets[i]],dtype=float)[keep]

        #    if i != 0:
        #        axnew = ax.twiny()
        #        axnew.hist(dist,alpha=0.8,bins=30)
        #        axnew.set_xlabel(column_to_label(targets[i]))
        #        axnew.spines["bottom"].set_position(("axes", 1.2))
        #        # Having been created by twinx, par2 has its frame off, so the line of its
        #        # detached spine is invisible.  First, activate the frame but make the patch
        #        # and spines invisible.
        #        make_patch_spines_invisible(axnew)
        #        # Second, show the right spine.
        #        axnew.spines["bottom"].set_visible(True)
        #    else:
        #        ax.hist(dist,alpha=0.8,bins=30)
        #        ax.set_xlabel(column_to_label(targets[i]))
        #        
        #    #ax.set_yscale('log')

        #plt.tight_layout(pad=0.5)
        #plt.show()

        #sys.exit()

    ########################################################################################
  

 
    ########################################################################################
    # ML model setup helper functions

    def get_train_test_split(self, X, y, test_size, random_state):
       
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            random_state = random_state)

        return X_train, X_test, y_train, y_test


    def get_kfold(self, X, nsplits, shuffle=True, random_state=0):

        kf = KFold(n_splits=nsplits, shuffle=shuffle, random_state=random_state)
        kf.get_n_splits(X)

        return kf

       
    def run_(self):
        # We should have separate function that executes the actual training
        # so that it can be distributed with multiprocessing
        pass

    def feature_reduction(self):
        # TODO:
        # https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
        pass

    ########################################################################################

    def run_NN_regressor(self,additional_holdout = {},
                                              predict_column = 'Heat_of_Formation_kJperMolH2',
                                              num_samples = 1,
                                              toplot=True,
                                              limlower=0,
                                              limupper=100,
                                              seed=0,
                                              keepfeatures=None,
                                              holdlower=True,
                                              holdupper=True):

        # Manual feature selection
        #keepfeatures=['mean_GSvolume_pa', 'dev_NdUnfilled', 'MeanIonicChar', 'mean_Electronegativity']
        #keepfeatures=['mean_MeltingT']
        #keepfeatures=['mean_GSvolume_pa','mean_CovalentRadius','dev_NdUnfilled','mean_Electronegativity']
        if keepfeatures is not None:
            self._features = retain_features(self._features,keepfeatures)
            self._features_HS = retain_features(self._features_HS,keepfeatures)

        # get all data samples        
        allX = np.array(self._features, dtype=float)
        ally = np.squeeze(np.array(self._database[predict_column],dtype=float))
       
        # double check removing any nan's but this has already been done
        # only useful if we wanted to try to train on MP attributes in which case we'd have 
        # to remove all the entries where we didn't find an entry on MP 
        mask = ~np.any(np.isnan(allX), axis=1)
        allX = allX[mask]
        ally = ally[mask]

        # Exculde samples as holdout if predict value is above/below a threshold
        keep_indices, holdout_indices = filter_by_predict_value(limlower,limupper,ally,holdlower,holdupper)

        # set up training data
        X = allX[keep_indices]
        y = ally[keep_indices]

        # setup holdout data
        Xhold = allX[holdout_indices]
        yhold = ally[holdout_indices]

        # setup more holdout data (i.e. if we want to test on info from different databases) 
        for key in additional_holdout.keys():
            features = additional_holdout[key]['features']
            database = additional_holdout[key]['database']
            this_Xhold = np.array(features)
            this_yhold = np.squeeze(np.array(database[predict_column],dtype=float))
            print(np.shape(this_Xhold))
            print(np.shape(this_yhold))

            additional_holdout[key]['Xhold']=this_Xhold
            additional_holdout[key]['yhold']=this_yhold
            additional_holdout[key]['allhold_pred']=[]
            additional_holdout[key]['allhold_mae']=[]

        # See how much the dataset decreases by when holding out
        print("Full dataset:")
        print("Features dim. : " + str(np.shape(allX)))
        print("Predict dim. : " + str(np.shape(ally)))

        print("Train/test dataset:")
        print("Features dim. : " + str(np.shape(X)))
        print("Predict dim. : " + str(np.shape(y)))

        print("Holdout dataset:")
        print("Features dim. : " + str(np.shape(Xhold)))
        print("Predict dim. : " + str(np.shape(yhold)))

    def run_gradient_boosting_regressor(self, additional_holdout = {},
                                              predict_column = 'Heat_of_Formation_kJperMolH2',
                                              num_samples = 1,
                                              toplot=True,
                                              limlower=0,
                                              limupper=100,
                                              seed=0,
                                              keepfeatures=None,
                                              holdlower=True,
                                              holdupper=True):
        """
        Overfitting and Gradient Boosting
        https://www.quora.com/How-do-you-correct-for-overfitting-for-a-Gradient-Boosted-Machine
        """


        # Manual feature selection
        #keepfeatures=['mean_GSvolume_pa', 'dev_NdUnfilled', 'MeanIonicChar', 'mean_Electronegativity']
        #keepfeatures=['mean_MeltingT']
        #keepfeatures=['mean_GSvolume_pa','mean_CovalentRadius','dev_NdUnfilled','mean_Electronegativity']
        if keepfeatures is not None:
            self._features = retain_features(self._features,keepfeatures)
            self._features_HS = retain_features(self._features_HS,keepfeatures)

        # get all data samples        
        allX = np.array(self._features, dtype=float)
        ally = np.squeeze(np.array(self._database[predict_column],dtype=float))
       
        # double check removing any nan's but this has already been done
        # only useful if we wanted to try to train on MP attributes in which case we'd have 
        # to remove all the entries where we didn't find an entry on MP 
        mask = ~np.any(np.isnan(allX), axis=1)
        allX = allX[mask]
        ally = ally[mask]

        # Exculde samples as holdout if predict value is above/below a threshold
        keep_indices, holdout_indices = filter_by_predict_value(limlower,limupper,ally,holdlower,holdupper)

        # set up training data
        X = allX[keep_indices]
        y = ally[keep_indices]

        # setup holdout data
        Xhold = allX[holdout_indices]
        yhold = ally[holdout_indices]

        # setup more holdout data (i.e. if we want to test on info from different databases) 
        for key in additional_holdout.keys():
            features = additional_holdout[key]['features']
            database = additional_holdout[key]['database']
            this_Xhold = np.array(features)
            this_yhold = np.squeeze(np.array(database[predict_column],dtype=float))
            print(np.shape(this_Xhold))
            print(np.shape(this_yhold))

            additional_holdout[key]['Xhold']=this_Xhold
            additional_holdout[key]['yhold']=this_yhold
            additional_holdout[key]['allhold_pred']=[]
            additional_holdout[key]['allhold_mae']=[]

        # See how much the dataset decreases by when holding out
        print("Full dataset:")
        print("Features dim. : " + str(np.shape(allX)))
        print("Predict dim. : " + str(np.shape(ally)))

        print("Train/test dataset:")
        print("Features dim. : " + str(np.shape(X)))
        print("Predict dim. : " + str(np.shape(y)))

        print("Holdout dataset:")
        print("Features dim. : " + str(np.shape(Xhold)))
        print("Predict dim. : " + str(np.shape(yhold)))
    
        # test/train setup 
        startseed=7
        seeds = np.arange(startseed,startseed+num_samples,1)
        modelstats = []
        seed=1
        test_size=0.1
        #test_size=0.5
        nsplits=int(np.ceil(1/test_size))
        kf = self.get_kfold(X, nsplits, True, seed)
        #for i in seeds:

        ncols=4
        fig, ax = plt.subplots(nrows=nsplits,ncols=ncols,figsize=(3.3*ncols,1.9*nsplits),
                               gridspec_kw={'width_ratios': [3,3,3,1]})
        it=0
        print("Figure is n x m = %d x %d"%(nsplits,ncols))
        print(ax[0,0])
        all_train_pred = []
        all_train_mae = []
        all_test_pred = []
        all_test_mae = []
        all_hold_pred = []
        all_hold_mae = []
        all_feature_importance=np.zeros(np.shape(self._features)[1])
        for train_index,test_index in kf.split(X):

            #random_state = seed

            #train_test_param = {'X'           : features,
            #                    'y'           : prediction,
            #                    'test_size'   : test_size,
            #                    'random_state': seed}
            #X_train, X_test, y_train, y_test = self.get_train_test_split(**train_test_param)
            #X_train, X_holdout, y_train, y_holdout = self.get_train_test_split(**train_test_param)

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            n_estimators = 15000
            #n_estimators = 2
            learning_rate = 0.0005
            max_depth = 4
            random_state = seed
            loss='ls'

            param = {
                        'n_estimators'  : n_estimators, 
                        'learning_rate' : learning_rate,
                        'max_depth'     : max_depth, 
                        'random_state'  : seed, 
                        'loss'          : loss,
                        'subsample'     : 0.75,
                        'alpha'         : 0.99
                    }

            # Fit GB model, NN, LASSO, Linear, or Kernel Ridge regressors
            est = GradientBoostingRegressor(**param).fit(X_train, y_train)
            #est = MLPRegressor(solver='lbfgs', alpha=1e-5,
            #                   hidden_layer_sizes=(64,64,64,64), random_state=seed).fit(X_train,y_train)
            #est = linear_model.Lasso(alpha=0.001).fit(X_train,y_train)            
            #est = linear_model.LinearRegression().fit(X_train,y_train)            
            #est = KernelRidge(alpha=0.01).fit(X_train, y_train)

            # How to do hyperparameter search
            # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
            #est = RandomForestRegressor(random_state=seed)
            #random_grid = get_random_RF_param_grid()            
            #rf_random = RandomizedSearchCV(estimator = est, param_distributions = random_grid, 
            #                               n_iter = 10, cv = 4, verbose=2, random_state=seed, 
            #                               n_jobs = -1)
            #rf_random.fit(X_train, y_train)
            #best_param = rf_random.best_params_

            #est = RandomForestRegressor(random_state=seed,**best_param)
            #est.fit(X_train, y_train)
            dump(est,'savedMLmodel.joblib')

            # evaluate model on the training set
            y_train_pred = est.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mre = compute_mean_relative_error(y_train, y_train_pred)
            all_train_pred.append((y_train, y_train_pred))
            all_train_mae.append(train_mae)
        
            # evaluate model on the test set
            y_test_pred = est.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mre = compute_mean_relative_error(y_test, y_test_pred)
            all_test_pred.append((y_test,y_test_pred))
            all_test_mae.append(test_mae)

            # evaluate model on the holdout set
            if len(yhold) != 0:
                yhold_pred = est.predict(Xhold)
                holdout_mse = mean_squared_error(yhold, yhold_pred)
                holdout_mae = mean_absolute_error(yhold, yhold_pred)
                holdout_mre = compute_mean_relative_error(yhold, yhold_pred)
                all_hold_pred.append((yhold, yhold_pred))
                all_hold_mae.append(holdout_mae)

            # evaluate model on any additional holdout sets
            for key in additional_holdout.keys():
                thisy = additional_holdout[key]['yhold']
                thisy_pred = est.predict(additional_holdout[key]['Xhold'])
                this_mae = mean_absolute_error(thisy, thisy_pred)
                print(this_mae)
                additional_holdout[key]['allhold_pred'].append((thisy, thisy_pred))
                additional_holdout[key]['allhold_mae'].append(this_mae)

            print(seed, train_mae, test_mae, test_mre) 
            modelstats.append([seed, train_mae, test_mae, test_mre])
        
            #print("Train set:")
            #print("Train set:\nMSE = %f, MAE = %f"%(train_mse, train_mae))
            #print("Test set:\nMSE = %f, MAE = %f"%(test_mse, test_mae))

            if toplot: 
                #fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,2.7))

                ax[it,0].scatter(np.array(y_train),est.predict(X_train),edgecolor='blue',
                                                                   linewidths=1,
                                                                   alpha=0.3,
                                                                   label="Train MAE = %.2f"%(train_mae))
                ax[it,0].set_xlabel(r"True %s"%column_to_label(predict_column))
                ax[it,0].set_ylabel(r"Model %s"%column_to_label(predict_column))
                draw_y_equals_x(ax[it,0])
                ax[it,0].legend(loc='best')
            
                
                SC = stats.spearmanr(np.array(y_test),est.predict(X_test))
                ax[it,1].scatter(np.array(y_test),est.predict(X_test),
                              edgecolor='blue',
                              linewidths=1,
                              alpha=0.3,
                              label="Test MAE (SC)= %.2f (%.2f)"%(test_mae,SC[0]))
                if len(yhold) != 0:
                    ax[it,1].scatter(np.array(yhold),est.predict(Xhold),
                                  edgecolor='red',
                                  linewidths=1,
                                  alpha=0.3,
                                  label="Holdout MAE = %.2f"%(holdout_mae))
                ax[it,1].set_xlabel(r"True %s"%column_to_label(predict_column))
                ax[it,1].set_ylabel(r"Model %s"%column_to_label(predict_column))
                draw_y_equals_x(ax[it,1])
                ax[it,1].legend(loc='best')

                #plt.tight_layout(pad=0.5)
                #plt.show()
                #plt.close()

                # Obtain learning curves
                test_score = None
                if test_score is not None:
                    #test_score = np.zeros((param['n_estimators'],), dtype=np.float64)
                    for i, y_pred in enumerate(est.staged_predict(X_test)):
                        test_score[i] = est.loss_(y_test, y_pred)


                    # Plot staged predicitions
                    #fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(3.3,2.6))
                    #print(i, ncols)
                    #ax[it,2].set_title('Deviance')
                    ax[it,2].plot(np.arange(param['n_estimators']) + 1, est.train_score_, 'b-',
                            label='Training Set Deviance')
                    ax[it,2].plot(np.arange(param['n_estimators']) + 1, test_score, 'r-',
                               label='Test Set Deviance')
                    ax[it,2].legend(loc='upper right')
                    ax[it,2].set_xlabel('Boosting Iterations')
                    ax[it,2].set_ylabel('Deviance')

        
                # Plot feature importance
                #fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(4,2.6))
                try: 
                    feature_importance = est.feature_importances_

                    # make importances relative to max importance
                    feature_importance = 100.0 * (feature_importance / feature_importance.max())
                    all_feature_importance += feature_importance
                    sorted_idx = np.argsort(feature_importance)
                    pos = np.arange(sorted_idx.shape[0]) + .5
                    maxdisplay = min(len(pos),8) # we only want to plot a max num of features

                    ax[it,3].barh(pos[-maxdisplay:], feature_importance[sorted_idx][-maxdisplay:], align='center')
                    ax[it,3].set_yticks(pos[-maxdisplay:])
                    # Latex doesn't like the _ so need to replace it
                    ticklabels = [feature.replace('_','\_')\
                                  for feature in self._features.columns[sorted_idx][-maxdisplay:]]
                    ax[it,3].set_yticklabels(ticklabels)
                    ax[it,3].set_xlabel('Relative Importance')
                    #ax[it,3].set_title('Variable Importance')
                    #plt.tight_layout(pad=0.1)
                    #plt.show()
                except:
                    pass

            it+=1

        plt.tight_layout(pad=0.1,h_pad=0.1)
        plt.show()
        plt.close()

        
        # Plot publication figure of combined k-fold train and test
        fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(4.5,2.3), sharey=True) 
        # plot combined k-fold test predictions
        for i in range(len(all_test_pred)):
            if i == 0:
                testlabel = r"$\langle$MAE$\rangle_{Test}$ = %.2f"%(np.average(all_test_mae))
                trainlabel = r"$\langle$MAE$\rangle_{Train}$ = %.2f"%(np.average(all_train_mae))
            else:
                testlabel = None
                trainlabel = None

            ax[0].scatter(all_train_pred[i][0],all_train_pred[i][1], edgecolor='blue',
                                                                color='blue',
                                                                linewidths=1,
                                                                alpha=0.2,
                                                                label=trainlabel)
            ax[1].scatter(all_test_pred[i][0],all_test_pred[i][1], edgecolor='blue',
                                                                color='blue',
                                                                linewidths=1,
                                                                alpha=0.2,
                                                                label=testlabel)

            # Just use the k-fold model to demonstrate importance, too much space to show for all k-folds
            #ax[2].barh(pos[-maxdisplay:], feature_importance[sorted_idx][-maxdisplay:], align='center')
            #ax[2].set_yticks(pos[-maxdisplay:])
            # Latex doesn't like the _ so need to replace it
            #ticklabels = [feature.replace('_','\_')\
            #              for feature in self._features.columns[sorted_idx][-maxdisplay:]]
            #ax[2].set_yticklabels(ticklabels)
            #ax[2].set_xlabel('Relative Importance')

        # Plot any holdout data created by exclusion from the train/test
        for i in range(len(all_hold_pred)):
            if i == 0:
                thislabel = r"Hold $\langle$MAE$\rangle$ = %.2f"%(np.average(all_hold_mae))
            else:
                thislabel = None

            ax[1].scatter(all_hold_pred[i][0],all_hold_pred[i][1], edgecolor='red',
                                                                color='red',
                                                                linewidths=1,
                                                                alpha=0.2,
                                                                label=thislabel)

        # Plot all additional holdout, i.e. additional datasources
        for key in additional_holdout.keys():
            for i in range(len(additional_holdout[key]['allhold_pred'])):
                if i == 0:
                    thislabel = r"%s $\langle$MAE$\rangle$ = %.2f"%(key, np.average(additional_holdout[key]['allhold_mae']))
                else:
                    thislabel = None

                ax[1].scatter(additional_holdout[key]['allhold_pred'][i][0], 
                           additional_holdout[key]['allhold_pred'][i][1],
                                                            edgecolor='green',
                                                            color='green',
                                                            linewidths=1,
                                                            alpha=0.3,
                                                            label=thislabel)
        draw_y_equals_x(ax[0])
        draw_y_equals_x(ax[1])
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[0].set_ylabel(r"Model %s"%column_to_label(predict_column))
        ax[0].set_xlabel(r"True %s"%column_to_label(predict_column))
        ax[1].set_xlabel(r"True %s"%column_to_label(predict_column))
        plt.tight_layout(pad=0.5)
        #plt.show()


        # Plot publication figure of average importances over all k-fold models
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2.9,2.3))
       
        #print(np.shape(all_feature_importance)) 
        all_feature_importance/=(it+1)
        sorted_idx = np.argsort(all_feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        maxdisplay = min(len(pos),8) # we only want to plot a max num of features

        ax.barh(pos[-maxdisplay:], all_feature_importance[sorted_idx][-maxdisplay:], align='center')
        ax.set_yticks(pos[-maxdisplay:])
        # Latex doesn't like the _ so need to replace it
        ticklabels = [feature.replace('_','\_')\
                      for feature in self._features.columns[sorted_idx][-maxdisplay:]]
        ax.set_yticklabels(ticklabels)
        ax.set_xlabel(r'$\langle$Relative Importance$\rangle$')
        plt.tight_layout(pad=0.5)
        #plt.show()


        # Plot the MAE of subsets of the data binned on the true predict value
        nbins=15
        concatenated = np.concatenate(all_test_pred,axis=1)
        hist, bin_edges = np.histogram(concatenated, bins=nbins)
        print(hist, bin_edges)
        binned_AEs = [[] for _ in range(len(hist))]
        for i in range(len(concatenated[0,:])):
            bin_ind=1

            while concatenated[0,i] > bin_edges[bin_ind]:
                bin_ind+=1

            #print(concatenated[0,i], bin_ind)
            binned_AEs[bin_ind-1].append(np.abs(concatenated[0,i]-concatenated[1,i]))


        allxs = np.array([bin_edges[i-1]+(bin_edges[i]-bin_edges[i-1])/2 for i in range(1,len(bin_edges))])
        allys = np.array([np.mean(data) for data in binned_AEs])

        print(~np.isnan(allys))
        allxs = allxs[~np.isnan(allys)]
        allys = allys[~np.isnan(allys)]


        print(allxs)
        print(allys)
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2.9,2.3))
        ax.hist(concatenated[0,:],color='blue',bins=nbins)
        ax.set_xlabel(r"True %s"%column_to_label(predict_column))
        ax.set_ylabel(r"Frequency in database")
        ax.tick_params(axis='y', colors='blue')
        ax.yaxis.label.set_color('blue')
    
        axtwin = ax.twinx()
        axtwin.plot(allxs, allys,c="orange",marker="s",markerfacecolor="white")
        axtwin.set_ylabel(r"$\langle$MAE$\rangle_{Test}$ within bin")
        axtwin.tick_params(axis='y', colors='orange')
        axtwin.yaxis.label.set_color('orange')

        twin_ylims = axtwin.get_ylim()
        axtwin.set_ylim((0,min(twin_ylims[1],15)))

        plt.tight_layout(pad=0.5)
        plt.show()
        

            
        
                                                            


        


if __name__ == "__main__":

    database     = sys.argv[1]
    headerlength = int(sys.argv[2])

    recalculate_HYDPARK = False
    run_training        = False

    recalculate_MP = False
    run_MP_analyze = True

    remove_nan_attr=['Heat_of_Formation_kJperMolH2',
                          #'Hydrogen_Weight_Percent',
                          'Temperature_oC',
                          'Pressure_Atmospheres_Absolute']

    remove_classes = []
    
    database = H2Data(database,remove_nan_attr,remove_classes,headerlength=headerlength)
    #database.clean_composition_formula("Mg2Ni.75Fe.25 (M)")
    #database.clean_composition_formula("Mg2(Hg2C.7) (M)")

    # Data/features for chemical compositions only, i.e. HYDPARK and HS database
    if recalculate_HYDPARK:
        database.prepare_database_and_features(save=True,
                                                dbname = "HYDPARK_database.pkl",
                                                featuresname = "HYDPARK_magpie_features.pkl")
        database.prepare_database_and_features_HS("HS/HS_database.csv",
                                                  save=True,
                                                  dbname="HS_database.pkl",
                                                  featuresname="HS_magpie_features.pkl")
    else:
        database.load_database_and_features("HYDPARK/HYDPARK_database.pkl", 
                                            "HYDPARK/HYDPARK_magpie_features.pkl")
        database.load_database_and_features_HS("HS/HS_database.pkl", 
                                               "HS/HS_magpie_features.pkl")


    # Peq varies by orders of magnitude when standardized to room temp, so 
    # our objective is to predict the natural log
    database.convert_eqlb_pressure_to_ln()

    # Analyze/Remove duplicate Composition Formula's in the database
    #database.analyze_duplicates()
    database.clean_duplicates(verbose=True)
    #database.plot_prediction_distribution(['Heat_of_Formation_kJperMolH2',
    #                                       'LnEquilibrium_Pressure_25C',
    #                                       'Entropy_of_Formation_kJperMolH2perK'])

    # Clean complex hydrides
    allcomplex = database._database.loc[database._database['Material_Class'] ==\
                                         'Complex','Composition_Formula']
    [database.clean_specific_composition(comp) for comp in allcomplex]


    # Execute the training
    if run_training:
        additional_holdout = {}#{'HS': {'database': database._database_HS, 
                               #        'features': database._features_HS}}

        # For seeing how model depends on feature exclusion
        remove_cols = [col for col in database._features.columns if col != 'mean_GSvolume_pa']
        #database._features= database._features.drop(remove_cols,axis=1)
        database._features= database._features.drop(['mean_GSvolume_pa'],axis=1)
        database._features= database._features.drop(['most_GSvolume_pa'],axis=1)

        #database.run_gradient_boosting_regressor(predict_column='LnEquilibrium_Pressure_25C',
        #                                         toplot=True,
        #                                         num_samples=1,
        #                                         limlower=-20,
        #                                         limupper=5,
        #                                         holdlower=False,
        #                                         holdupper=False)
        database.run_gradient_boosting_regressor(additional_holdout=additional_holdout,
                                                 predict_column='Heat_of_Formation_kJperMolH2',
                                                 toplot=True,
                                                 num_samples=1,
                                                 limlower=0,
                                                 limupper=100,
                                                 holdlower=False,
                                                 holdupper=False)
        #database.run_gradient_boosting_regressor(additional_holdout=additional_holdout,
        #                                         predict_column='Entropy_of_Formation_kJperMolH2perK',
        #                                         toplot=True,
        #                                         num_samples=1,
        #                                         limlower=0,
        #                                         limupper=0.2,
        #                                         holdlower=False,
        #                                         holdupper=False)
    database.model = load("savedMLmodel.joblib")
        


    ###############################################################################################
    # Now that we have our ML model and understand the feature importance of mean_GSVolume_pa
    # we can further investigate each structure in more detail by querying the Materials Project
    # for structural/electronic structure information not captured by the Magpie descriptors


    if run_MP_analyze:

        # For the cleaned database, get Materials Project (MP) structure features
        if recalculate_MP:
            MP_features = database.prepare_MP_features(database._database, save=True, fname = "HYDPARK_MP_features.pkl")
        else:
            MP_features = database.load_MP_features("HYDPARK/HYDPARK_MP_features.pkl")

        # Clean MP structural features/derive new ones
        cleaned_MP_features = database.clean_MP_features(MP_features)

        # Compute any derived MP features
        cleaned_MP_features = database.compute_MP_volume_ps(cleaned_MP_features)
        #cleaned_MP_features = database.compute_MP_volume_descr_1()

        # Concatenate MP features to all features
        feature_list = ['energy_per_atom','formation_energy_per_atom','nsites','density','volume','volume_ps','empty_volume_ps']
        database.attach_MP_features_to_all(cleaned_MP_features, feature_list) 

        # Here we can remove/modify any structures we know to be problematic after more closely
        # investigating the literature references in HYDPARK
        database.clean_specific_composition('CeFe5')
        database.clean_specific_composition('GdFe2')
        database.clean_specific_composition('GdNi2')
        database.clean_specific_composition('GdMn2')
        database.manual_modify_thermo('Er6Fe23',50.4, .148)
        database.manual_modify_thermo('Ho6Fe23',51.7, .138)
        database._database['Entropy_of_Formation_kJperMolH2perK']=\
            database._database['Entropy_of_Formation_kJperMolH2perK']*1000
        allcomplex = database._database.loc[database._database['Material_Class'] ==\
                                            'Complex','Composition_Formula']
        [database.clean_specific_composition(comp) for comp in allcomplex]

        # TODO fix this: Write final dataset for easy manual inspection of data 
        with open("HYDPARK/HYDPARK_database_cleaned.csv","w") as f:
            newdf = pd.concat([database._database, database._features],sort=False, axis=1)
            newdf.to_csv(f)
        #bargraph_on_class(database._database)

        # Visualize the dependence of equilibrium pressure on mean elemental volume
        #feat='mean_GSvolume_pa'
        limlower2=-20
        #feat='empty_volume_ps'
        #plot_feature_vs_feature(database._features, feat,
        #                        database._database, 'LnEquilibrium_Pressure_25C',
        #                        cluster=False,
        #                        limlower2=limlower2,
        #                        limupper2=5,
        #                        display_SC=True)

        # Visualize the entropy/enthalpy trade off from individual normalized contributions
        # to lnPeqo 
        if False:
            plot_feature_vs_feature(\
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    limlower1=0,
                                    limupper1=200,
                                    limupper2=100,
                                    cluster=False,
                                    display_SC=True,
                                    specialdelHvsdelS=True,
                                    figsize=(2.2,2.6))

        # Visualize entropy/enthalpy tradeoff with contours of constant lnPeqo
        if False:
            plot_feature_vs_feature(\
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    limlower1=0,
                                    limupper1=200,
                                    limlower2=0,
                                    limupper2=100,
                                    cluster=False,
                                    specialdelHvsdelSv2=True,
                                    figsize=(3.3,2.3))

        # Visualize entropy/enthalpy tradeoff as a function of material class
        if False:
            plot_feature_vs_feature(\
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    limlower1=0,
                                    limupper1=200,
                                    limlower2=0,
                                    limupper2=100,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(5,3.5))


        # Visualize the dependence of enthalpy on nu_pa^Magpie colored by material class
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'mean_GSvolume_pa',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    limlower2=0,
                                    limupper2=100,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(5,3.5))
        # Visualize the dependence of enthalpy on nu_pa^Magpie colored by electronegativity
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'mean_GSvolume_pa',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    df3 = database._features, feature3 = 'most_Electronegativity',
                                    limlower2=0,
                                    limupper2=100,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(5,3.5))
        # Visualize the dependence of lnPeqo on nu_pa^Magpie colored by material class
        if True:
            plot_feature_vs_feature(\
                                    database._features, 'mean_GSvolume_pa',
                                    database._database, 'LnEquilibrium_Pressure_25C',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    limlower2=limlower2,
                                    limupper2=5,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(5,3.5))
        # Visualize the dependence of entropy on mean space group number colored by material class
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'mean_SpaceGroupNumber',
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    limlower1=180,
                                    limlower2=0,
                                    limupper2=200,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(5,3.5))

        if False:
            plot_feature_vs_feature(\
                                    #database._features, 'mean_GSvolume_pa',
                                    database._features, 'MeanIonicChar',
                                    #database._features, 'formation_energy_per_atom',
                                    #database._database, 'Heat_of_Formation_kJperMolH2',
                                    database._database, 'LnEquilibrium_Pressure_25C',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    #limlower2=0,
                                    #limupper2=0.075,
                                    limupper1=0.075,
                                    limlower2=-20,
                                    limupper2=5,
                                    cluster=False,
                                    figsize=(3.3,2.6))
        if False:
            plot_feature_vs_feature(\
                                    #database._features, 'mean_GSvolume_pa',
                                    database._features, 'mean_MeltingT',
                                    #database._database, 'Heat_of_Formation_kJperMolH2',
                                    database._database, 'LnEquilibrium_Pressure_25C',
                                    df3 = database._database, feature3 = 'Material_Class',
                                    #limlower2=0,
                                    #limupper2=100,
                                    limlower2=-20,
                                    limupper2=5,
                                    cluster=False,
                                    display_SC=True,
                                    figsize=(3.3,2.6))


        # Enthalpy vs nu_pa^MP, for outlier identification in the nu_pa < 17 range
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'volume_ps',
                                    database._database, 'Heat_of_Formation_kJperMolH2',
                                    df3 = database._features, feature3 = 'mean_CovalentRadius',
                                    limlower2=0,
                                    limupper2=100,
                                    cluster=False,
                                    )
                                    #display_SC=True)

        # Entropy vs nu_pa^MP, not an important descriptor
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'mean_GSvolume_pa',
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    limlower2 = 0,
                                    limupper2 = 200,
                                    cluster=False,
                                    display_SC=True)


        # Entropy vs mean space group number, the most important descriptor
        if False:
            plot_feature_vs_feature(\
                                    database._features, 'mean_SpaceGroupNumber',
                                    database._database, 'Entropy_of_Formation_kJperMolH2perK',
                                    limlower1=180,
                                    limupper2=300,
                                    cluster=False,
                                    display_SC=True)

        
        # Visualize the Magpie (mean elemental volume) vs MP volume / site
        if False:
            plot_feature_vs_feature(database._features, 'mean_GSvolume_pa',
                                    database._features, 'volume_ps',
                                    cluster=False,
                                    display_SC=True,
                                    drawyequalsx=True,
                                    figsize=(3.3,2.6))


        # Plot fig: Peq_vs_volumes.pdf
        if False:
            feat='volume'
            plot_feature_vs_feature(database._features, feat,
                                    database._database, 'LnEquilibrium_Pressure_25C',
                                    limlower2=limlower2,
                                    limupper2=5,
                                    cluster=True,
                                    display_SC=False,
                                    special_volume_fig=True,
                                    figsize=(2.5,2.0))
            # Plot fig: element_vs_site_volumes.pdf
            plot_prediction_vs_custom_volumes(database._features,
                                    database._database,
                                    limlower2=limlower2,
                                    limupper2=5,
                                    display_SC=True)

            # Plot fig: delH_vs_Magpie_MP_vsites.pdf
            plot_delHprediction_vs_custom_volumes(database._features,
                                                   database._database,
                                                   limlower2=0,
                                                   limupper2=100)


        # Validate on U Ni 5
        m = MagpieServer()
        batch = ["UNi5","U.25La.75Ni5","U.5La.5Ni5","U.25La.75Ni5"]
        magpie_attr  = m.generate_attributes(method,batch)
        predicitions = database.model.predict(magpie_attr)
        print(predictions)

