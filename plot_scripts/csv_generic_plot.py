import matplotlib
from matplotlib.ticker import LogLocator
# matplotlib.use('TkAgg')  # or 'Agg' for non-interactive backend
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import seaborn as sns
import matplotlib.patches as mpatches

datasets = [{
        'name'  : 'DGEMM',
        'ID'    : 'ecc21d0a-112a-4b52-8cdd-6aca80adde93',
        'cond'  : "halide_hpx_for",
    }, {
        'name'  : 'K-Means',
        'ID'    : 'faf17535-2f66-4621-995f-49c7dbd84e8b',
        'cond'  : "_phylanx$0___add$0_1$45$8",
    }, {
        'name'  : 'LULESH',
        'ID'    : 'a01b2607-32a6-4435-a2ae-20a4d227e5fd',
        'cond'  : "run_on_completed_on_new_thread",
    }, {
        'name'  : 'LRA',
        'ID'    : 'f4e2fdfa-893e-4f13-bac8-e9fbbdf40c1f',
        'cond'  : "run_on_completed_on_new_thread",
    }, {
        'name'  : 'Fibonacci',
        'ID'    : 'ae63b22a-66a3-4e92-ae49-b8206d8a0e7b',
        'cond'  : "run_on_completed_on_new_thread",
    }, {
        'name'  : 'Synthesized',
        'ID'    : '908fc737-2cc7-41d8-8281-7dd9e83155ff',
        'cond'  : "_phylanx$0___add$0_1$45$8",
}]
alg = ['eseman_kdt', 'eseman_kdt_twod', 'agglomerative_clustering', 'summed_area_table', 'db_duck_raw', 'db_duck_sketch', 'db_duck_min_max']
alg_code = ['ESeMan-1DKDT', 'ESeMan-KDT', 'ESeMan-Agg', 'Summed Area Table', 'Naive', 'Statistical Sub-sampling', 'M4 Optimization']
query = ['window', 'cond', 'window_picture', 'cond_picture']
base_directory = '/mnt/d/LDAV25Data/traveler-benchmarks'
qn = {'window': 'Range Query', 'cond': 'Conditiaonal Range Query'}
color = ['#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255', '#332288']
# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
pd.options.display.float_format = "{:,.2f}".format

def generateTimeDistribution(dataset_name, query_name):
    # ds_name = 'summed_area_table'
    ds_name = 'kd_tree'
    sat_data = pd.read_csv(dataset_name + "/" + ds_name + "_" + query_name + "_merged.csv")
    sat_data["Time between DS and API"] = sat_data["Post Processing Time(micros)"] - sat_data["API Fetch Time(micros)"]
    sat_data["DS API percentage"] = (sat_data["Time between DS and API"] / sat_data["total drawing time (micros)"]) * 100.0

    sat_data["Drawing only Time"] = sat_data["total drawing time (micros)"] - sat_data["Post Processing Time(micros)"]
    sat_data["Drawing percentage"] = (sat_data["Drawing only Time"] / sat_data["total drawing time (micros)"]) * 100.0

    sat_data["API percentage"] = (sat_data["Post Processing Time(micros)"] / sat_data["total drawing time (micros)"]) * 100.0

    sat_data.round(2)
    print(ds_name)
    print(sat_data[['total drawing time (micros)', 'Time between DS and API', 'DS API percentage', 'Drawing only Time', 'Drawing percentage', 'API percentage']])
    return

def checkIfResultsGenerated(ds, query_name, alg_name):
    file_path = base_directory + "/" + ds['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
    if query_name.endswith('_picture'):
        query_name_base = query_name.replace('_picture', '')
        cond_value = ''
        if query_name.startswith('cond'):
            cond_value = '_' + ds['cond']
        file_path = base_directory + "/" + ds['ID'] + "/" + ds['ID'] + "_" + query_name_base + "_" + alg_name + "_1" + cond_value + "_gantt.png"
        try:
            with open(file_path, 'rb') as f:
                return True
        except FileNotFoundError:
            return False
    else:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 10:
                return True
        except FileNotFoundError:
            return False
    return False

def resultChecker():
    for dataset in datasets:
        for query_name in query:
            for alg_name in alg:
                if checkIfResultsGenerated(dataset, query_name, alg_name) is False:
                    print(f"Results not generated for {dataset['name']} - {query_name} - {alg_name}")

def drawPlot(query_name, metric_type, is_first):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Initialize a dictionary to store data for each algorithm
    alg_data = {a: [] for a in alg}
    dataset_names = []

    # Collect data for each dataset and algorithm
    for i, dataset in enumerate(datasets):
        dataset_names.append(dataset['name'])
        for alg_name in alg:
            if (alg_name == 'eseman_kdt' or alg_name == 'eseman_kdt_twod' or alg_name == 'agglomerative_clustering') and metric_type == 'data_fetch':
                metric_column = 'ds query time (micros)'
            elif metric_type == 'data_fetch':
                metric_column = 'Post Processing Time(micros)'
            elif metric_type == 'inp':
                metric_column = 'inp (micros)'
            elif metric_type == 'total_drawing':
                metric_column = 'total drawing time (micros)'
            elif metric_type == 'ssim':
                metric_column = 'ssim'

            if not checkIfResultsGenerated(dataset, query_name, alg_name):
                alg_data[alg_name].append(np.nan)
                continue
            try:
                file_path = ""
                df = pd.DataFrame()
                if metric_type == 'data_fetch':
                    file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
                    if alg_name.startswith('db'):
                        df = pd.read_csv(file_path, skiprows=1, skipfooter=3, engine='python')
                    else:
                        df = pd.read_csv(file_path, skipfooter=3, engine='python')
                elif metric_type == 'inp' or metric_type == 'total_drawing' or metric_type == 'ssim':
                    file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                    df = pd.read_csv(file_path)
                
                if is_first:
                    df = df.head(1)  # Take first element only
                else:
                    df = df.tail(10)  # Take last 10 elements
                
                
                if metric_column in df.columns:
                    alg_data[alg_name].append(df[metric_column].mean())
                else:
                    alg_data[alg_name].append(np.nan)
            except:
                alg_data[alg_name].append(np.nan)

    # Example: plot a normalized line (random or fixed values between 0 and 1)
    # secondary_values = np.linspace(0.2, 0.9, len(dataset_names))
    sism_values = {a: [] for a in alg}
    for i, dataset in enumerate(datasets):
        # dataset_names.append(dataset['name'])
        for alg_name in alg:
            metric_column = 'ssim'

            if not checkIfResultsGenerated(dataset, query_name, alg_name):
                sism_values[alg_name].append(np.nan)
                continue
            try:
                file_path = ""
                df = pd.DataFrame()
                file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                df = pd.read_csv(file_path)
                
                if is_first:
                    df = df.head(1)  # Take first element only
                else:
                    df = df.tail(10)  # Take last 10 elements
                
                
                if metric_column in df.columns:
                    sism_values[alg_name].append(df[metric_column].mean())
                else:
                    sism_values[alg_name].append(np.nan)
            except:
                sism_values[alg_name].append(np.nan)

    
    # Plot a line for each algorithm
    line_styles = [':', '-', '-', '-.', ':', '--']  # Define different line styles
    markers = ['o', 's', '*', 'v', 'D', '^']  # Define different markers
    bar_width = 0.12
    group_gap = 1#1.25
    x = np.arange(len(dataset_names))
    # Use a colormap for distinct colors
    # cmap = plt.get_cmap('tab20c')

    # Define hatch patterns for each algorithm
    hatch_patterns = ['/', '\\', '|', '-', 'o', 'x']

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # Main bar chart (top)
    ax_main = fig.add_subplot(gs[0])
    for i, alg_name in enumerate(alg):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        bar_alpha = 0.5
        ax_main.bar(
            x * group_gap + i * bar_width,
            alg_data[alg_name],
            width=bar_width,
            label=alg_code[i],
            color=color[i % len(color)],
            hatch=hatch,
            edgecolor='black',
            alpha=bar_alpha,
            linewidth=1.5
        )
    ax_main.set_xticks(x * group_gap + bar_width * (len(alg) - 1) / 2)
    ax_main.set_xticklabels(dataset_names, fontsize=20)

    ax_main.axhline(y=100000, color=color[len(color)-1], linestyle='--', label='Interactivity Threshold')

    # ax_main.set_xlabel("Datasets")
    if metric_type == 'data_fetch':
        # ax_main.set_title(f"Data Fetch Time Comparison - {qn[query_name]}")
        ax_main.set_ylabel("Average Data Fetch Time (s)")
    elif metric_type == 'inp':
        # ax_main.set_title(f"Interaction to Next Point Comparison - {qn[query_name]}")
        ax_main.set_ylabel("Average INP Time (s)")
    elif metric_type == 'ssim':
        # ax_main.set_title(f"SSIM Comparison - {qn[query_name]}")
        ax_main.set_ylabel("SSIM")
        ax_main.set_ylim(0, 1)
    elif metric_type == 'total_drawing':
        # ax_main.set_title(f"Total Drawing Time Comparison - {qn[query_name]}")
        if is_first:
            ax_main.set_ylabel("First Drawing Time (s)")
        else:
            ax_main.set_ylabel("Average Drawing Time (s)")
    
    # Increase the number of yticks for better granularity
    # ax_main.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))

    ax_main.set_axisbelow(True)
    ax_main.grid(True)#, which="both")
    fig.subplots_adjust(top=0.8)
    ax_main.legend(
        fontsize=18, ncol=2, loc='upper center',
        bbox_to_anchor=(0.5, 1.40), frameon=False
    )
    ax_main.set_yscale('log')
    ax_main.yaxis.set_major_formatter(lambda x, p: f'{x/1000000:.2f}')
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax_main.xaxis.label.set_size(20)
    ax_main.yaxis.label.set_size(20)
    # ax_main.minorticks_on()
    # ax_main.set_title(ax_main.get_title(), fontsize=16)

    ax_main.set_yticks([x * 1000000 for x in [0.0002, 0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 32.0, 70.0]])
    ax_main.set_yticklabels(['0.2m','1m', '5m', 0.02, 0.1, 0.5,2.0,8.0, 32.0, 70.0])

    # SSIM bar chart (bottom)
    ax_ssim = fig.add_subplot(gs[1], sharex=ax_main)
    ssim_bar_width = bar_width
    for i, alg_name in enumerate(alg):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        ax_ssim.bar(
            x * group_gap + i * ssim_bar_width,
            1 - np.array(sism_values[alg_name]),
            width=ssim_bar_width,
            label=alg_code[i],
            color=color[i % len(color)],
            hatch=hatch,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.0
        )
    ax_ssim.text(
        0.02, 0.05, "Longer bars correspond to higher deviation from visual accuracy.",
        transform=ax_ssim.transAxes,
        fontsize=18,
        color='red',
        ha='left',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_ssim.set_ylabel("1 - SSIM")
    ax_ssim.yaxis.label.set_size(20)
    ax_ssim.set_ylim(0.7, 0)
    # ax_ssim.set_yscale('log')
    ax_ssim.set_xticks(x * group_gap + bar_width * (len(alg) - 1) / 2)
    ax_ssim.set_xticklabels(dataset_names, fontsize=16)
    ax_ssim.set_xlabel("Datasets")
    ax_ssim.set_axisbelow(True)
    ax_ssim.grid(True)
    # ax_ssim.legend(fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    ax_ssim.xaxis.label.set_size(20)
    plt.setp(ax_ssim.get_yticklabels(), fontsize=20)

    # plt.tight_layout()
    if is_first:
        plt.savefig(f"{query_name}_{metric_type}_first_comparison.png")
    else:
        plt.savefig(f"{query_name}_{metric_type}_comparison.png")
    plt.close()

def drawingFetchingComp(query_name):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Initialize a dictionary to store data for each algorithm
    alg_data = {a: [] for a in alg}
    dataset_names = []

    # Collect data for each dataset and algorithm
    for i, dataset in enumerate(datasets):
        dataset_names.append(dataset['name'])
        for alg_name in alg:
            pre_df = 0
            percent_increase = 0.0
            for metric_type in ['data_fetch', 'total_drawing']:
                if metric_type == 'data_fetch':
                    pre_df = 0
                if (alg_name == 'eseman_kdt' or alg_name == 'eseman_kdt_twod' or alg_name == 'agglomerative_clustering') and metric_type == 'data_fetch':
                    metric_column = 'ds query time (micros)'
                elif metric_type == 'data_fetch':
                    metric_column = 'Post Processing Time(micros)'
                elif metric_type == 'inp':
                    metric_column = 'inp (micros)'
                elif metric_type == 'total_drawing':
                    metric_column = 'total drawing time (micros)'
                elif metric_type == 'ssim':
                    metric_column = 'ssim'

                if not checkIfResultsGenerated(dataset, query_name, alg_name):
                    alg_data[alg_name].append(np.nan)
                    continue
                try:
                    file_path = ""
                    df = pd.DataFrame()
                    if metric_type == 'data_fetch':
                        file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
                        if alg_name.startswith('db'):
                            df = pd.read_csv(file_path, skiprows=1, skipfooter=3, engine='python')
                        else:
                            df = pd.read_csv(file_path, skipfooter=3, engine='python')
                    elif metric_type == 'inp' or metric_type == 'total_drawing' or metric_type == 'ssim':
                        file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                        df = pd.read_csv(file_path)

                    df = df.tail(10)  # Take last 10 elements
                    
                    
                    if metric_column in df.columns:
                        if metric_type == 'total_drawing':
                            c_df = df[metric_column].mean()
                            if c_df < pre_df:
                                print(f"Warning: Total drawing time less than data fetch time for {dataset['name']} - {alg_name}, c_df: {c_df}, pre_df: {pre_df}, diff: {pre_df - c_df}")
                            # if c_df > pre_df:
                                # compute percent increase from data fetch (pre_df) to total drawing (c_df)
                            if pre_df != 0:
                                percent_increase = ((c_df - pre_df) / pre_df) * 100.0
                                # # set pre_df so the existing append (pre_df / df[metric_column].mean()) yields the percent increase
                                # pre_df = percent_increase * c_df
                        else:
                            pre_df = df[metric_column].mean()

                    
                    # if alg_name == 'summed_area_table' and dataset['name'] == 'DGEMM':
                    #     print(df)
                except:
                    pass
                #     alg_data[alg_name].append(np.nan)
            alg_data[alg_name].append(percent_increase)



def totalDrawingFetchingPercentage(query_name):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Initialize a dictionary to store data for each algorithm
    alg_data = {a: [] for a in alg}
    pi_values = {a: [] for a in alg}
    dataset_names = []

    # Collect data for each dataset and algorithm
    for i, dataset in enumerate(datasets):
        dataset_names.append(dataset['name'])
        for alg_name in alg:
            pre_df = 0
            percent_increase = 0.0
            for metric_type in ['data_fetch', 'total_drawing']:
                if metric_type == 'data_fetch':
                    pre_df = 0

                if (alg_name == 'eseman_kdt' or alg_name == 'eseman_kdt_twod' or alg_name == 'agglomerative_clustering') and metric_type == 'data_fetch':
                    metric_column = 'ds query time (micros)'
                elif metric_type == 'data_fetch':
                    metric_column = 'Post Processing Time(micros)'
                elif metric_type == 'inp':
                    metric_column = 'inp (micros)'
                elif metric_type == 'total_drawing':
                    metric_column = 'total drawing time (micros)'
                elif metric_type == 'ssim':
                    metric_column = 'ssim'

                if not checkIfResultsGenerated(dataset, query_name, alg_name):
                    if metric_type == 'total_drawing':
                        alg_data[alg_name].append(np.nan)
                    continue
                try:
                    file_path = ""
                    df = pd.DataFrame()
                    if metric_type == 'data_fetch':
                        file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
                        if alg_name.startswith('db'):
                            df = pd.read_csv(file_path, skiprows=1, skipfooter=3, engine='python')
                        else:
                            df = pd.read_csv(file_path, skipfooter=3, engine='python')
                    elif metric_type == 'inp' or metric_type == 'total_drawing' or metric_type == 'ssim':
                        file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                        df = pd.read_csv(file_path)

                    df = df.tail(10)  # Take last 10 elements
                    
                    if metric_type == 'total_drawing':
                        if metric_column in df.columns:
                            alg_data[alg_name].append(df[metric_column].mean())
                        else:
                            alg_data[alg_name].append(np.nan)

                    if metric_column in df.columns:
                        if metric_type == 'total_drawing':
                            c_df = df[metric_column].mean()
                            # if c_df < pre_df:
                            #     print(f"Warning: Total drawing time less than data fetch time for {dataset['name']} - {alg_name}, c_df: {c_df}, pre_df: {pre_df}, diff: {pre_df - c_df}")
                            # if c_df > pre_df:
                                # compute percent increase from data fetch (pre_df) to total drawing (c_df)
                            if pre_df != 0:
                                percent_increase = ((c_df - pre_df) / pre_df) * 100.0
                                # # set pre_df so the existing append (pre_df / df[metric_column].mean()) yields the percent increase
                                # pre_df = percent_increase * c_df
                        else:
                            pre_df = df[metric_column].mean()
                except:
                    if metric_type == 'total_drawing':
                       alg_data[alg_name].append(np.nan)
            pi_values[alg_name].append(percent_increase)
    
    # Plot a line for each algorithm
    line_styles = [':', '-', '-', '-.', ':', '--']  # Define different line styles
    markers = ['o', 's', '*', 'v', 'D', '^']  # Define different markers
    bar_width = 0.12
    group_gap = 1#1.25
    x = np.arange(len(dataset_names))
    # Use a colormap for distinct colors
    # cmap = plt.get_cmap('tab20c')

    # Define hatch patterns for each algorithm
    hatch_patterns = ['/', '\\', '|', '-', 'o', 'x']

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    # Main bar chart (top)
    ax_main = fig.add_subplot(gs[0])
    for i, alg_name in enumerate(alg):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        bar_alpha = 0.5
        ax_main.bar(
            x * group_gap + i * bar_width,
            alg_data[alg_name],
            width=bar_width,
            label=alg_code[i],
            color=color[i % len(color)],
            hatch=hatch,
            edgecolor='black',
            alpha=bar_alpha,
            linewidth=1.5
        )
    ax_main.set_xticks(x * group_gap + bar_width * (len(alg) - 1) / 2)
    ax_main.set_xticklabels(dataset_names, fontsize=20)

    ax_main.axhline(y=100000, color=color[len(color)-1], linestyle='--', label='Interactivity Threshold')

    # ax_main.set_xlabel("Datasets")
    if metric_type == 'data_fetch':
        # ax_main.set_title(f"Data Fetch Time Comparison - {qn[query_name]}")
        ax_main.set_ylabel("Average Data Fetch Time (s)")
    elif metric_type == 'inp':
        # ax_main.set_title(f"Interaction to Next Point Comparison - {qn[query_name]}")
        ax_main.set_ylabel("Average INP Time (s)")
    elif metric_type == 'ssim':
        # ax_main.set_title(f"SSIM Comparison - {qn[query_name]}")
        ax_main.set_ylabel("SSIM")
        ax_main.set_ylim(0, 1)
    elif metric_type == 'total_drawing':
        ax_main.set_ylabel("Average Drawing Time (s)")
    
    # Increase the number of yticks for better granularity
    # ax_main.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))

    ax_main.set_axisbelow(True)
    ax_main.grid(True)#, which="both")
    fig.subplots_adjust(top=0.8)
    ax_main.legend(
        fontsize=18, ncol=2, loc='upper center',
        bbox_to_anchor=(0.5, 1.40), frameon=False
    )
    ax_main.set_yscale('log')
    ax_main.yaxis.set_major_formatter(lambda x, p: f'{x/1000000:.2f}')
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax_main.xaxis.label.set_size(20)
    ax_main.yaxis.label.set_size(20)
    # ax_main.minorticks_on()
    # ax_main.set_title(ax_main.get_title(), fontsize=16)

    ax_main.set_yticks([x * 1000000 for x in [0.0002, 0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 32.0, 70.0]])
    ax_main.set_yticklabels(['0.2m','1m', '5m', 0.02, 0.1, 0.5,2.0,8.0, 32.0, 70.0])
    # ax_main.set_yticks([x * 1000000 for x in [0.02, 0.1, 0.5, 2.0, 8.0, 32.0, 70.0]])
    # ax_main.set_yticklabels([0.02, 0.1, 0.5,2.0,8.0, 32.0, 70.0])

    # SSIM bar chart (bottom)
    ax_ssim = fig.add_subplot(gs[1], sharex=ax_main)
    ssim_bar_width = bar_width
    for i, alg_name in enumerate(alg):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        ax_ssim.bar(
            x * group_gap + i * ssim_bar_width,
            np.array(pi_values[alg_name]),
            width=ssim_bar_width,
            label=alg_code[i],
            color=color[i % len(color)],
            hatch=hatch,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.0
        )
    ax_ssim.text(
        0.01, 0.82, "No bar for percent decrease",
        transform=ax_ssim.transAxes,
        fontsize=16,
        color='red',
        ha='left',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax_ssim.set_ylabel("% Increase\nFrom Fetch")
    ax_ssim.yaxis.label.set_size(20)
    ax_ssim.set_yscale('log')
    # ax_ssim.set_ylim(1, 100000)
    ax_ssim.set_yticks([x for x in [1, 10, 100, 1000, 10000, 100000]])
    ax_ssim.set_yticklabels([1, 10, r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax_ssim.set_xticks(x * group_gap + bar_width * (len(alg) - 1) / 2)
    ax_ssim.set_xticklabels(dataset_names, fontsize=16)
    ax_ssim.set_xlabel("Datasets")
    ax_ssim.set_axisbelow(True)
    ax_ssim.grid(True)
    # ax_ssim.legend(fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    ax_ssim.xaxis.label.set_size(20)
    plt.setp(ax_ssim.get_yticklabels(), fontsize=20)

    # plt.tight_layout()
    plt.savefig(f"{query_name}_total_drawing_breakdown.png")
    plt.close()



def dataTester(query_name, metric_type, is_first):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Initialize a dictionary to store data for each algorithm
    alg_data = {a: [] for a in alg}
    dataset_names = []

    # Collect data for each dataset and algorithm
    for i, dataset in enumerate(datasets):
        dataset_names.append(dataset['name'])
        for alg_name in alg:
            if (alg_name == 'eseman_kdt' or alg_name == 'eseman_kdt_twod' or alg_name == 'agglomerative_clustering') and metric_type == 'data_fetch':
                metric_column = 'ds query time (micros)'
            elif metric_type == 'data_fetch':
                metric_column = 'Post Processing Time(micros)'
            elif metric_type == 'inp':
                metric_column = 'inp (micros)'
            elif metric_type == 'total_drawing':
                metric_column = 'total drawing time (micros)'
            elif metric_type == 'ssim':
                metric_column = 'ssim'

            if not checkIfResultsGenerated(dataset, query_name, alg_name):
                alg_data[alg_name].append(np.nan)
                continue
            try:
                file_path = ""
                df = pd.DataFrame()
                if metric_type == 'data_fetch':
                    file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
                    if alg_name.startswith('db'):
                        df = pd.read_csv(file_path, skiprows=1, skipfooter=3, engine='python')
                    else:
                        df = pd.read_csv(file_path, skipfooter=3, engine='python')
                elif metric_type == 'inp' or metric_type == 'total_drawing' or metric_type == 'ssim':
                    file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                    df = pd.read_csv(file_path)
                
                if is_first:
                    df = df.head(1)  # Take first element only
                else:
                    df = df.tail(10)  # Take last 10 elements
                
                
                if metric_column in df.columns:
                    alg_data[alg_name].append(df[metric_column].mean())
                else:
                    alg_data[alg_name].append(np.nan)
                if alg_name == 'summed_area_table' and dataset['name'] == 'DGEMM':
                    print(df)
            except:
                alg_data[alg_name].append(np.nan)



def compare_images(imageA, imageB):
    # Load the two PNG images (convert to grayscale for SSIM)
    image1 = imread(imageA, as_gray=True)
    image2 = imread(imageB, as_gray=True)

    # Compute SSIM
    score, diff = ssim(image1, image2, full=True, data_range=1.0)
    return score

def drawSSIMSmallMultiple():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize a dictionary to store data for each algorithm
    dataset_names = []
    alg_name = 'eseman_kdt'
    query_name = 'window'
    hrds = ['1', '2', '4', '8', '16', '32', '64']

    ssim_list = {}
    fetch_list = {}
    # # Collect data for each dataset and algorithm
    # for i, dataset in enumerate(datasets):
    #     dataset_names.append(dataset['name'])
    #     ssim_list[dataset['name']] = []
    #     fetch_list[dataset['name']] = []
        
    #     for hr in hrds:
    #         if not checkIfResultsGenerated(dataset, query_name, alg_name):
    #             ssim_list[dataset['name']].append(np.nan)
    #             fetch_list[dataset['name']].append(np.nan)
    #             continue
    #         try:
    #             df = pd.DataFrame()
    #             hrs = "" if hr == '1' else hr + "_"
    #             file_path = base_directory + "/" + hrs + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
    #             df = pd.read_csv(file_path, skipfooter=3, engine='python')
    #             df = df.tail(10)
    #             fetch_list[dataset['name']].append(df['ds query time (micros)'].mean())

    #             cssim = 0
    #             for i in range(11, 21):
    #                 base_png = base_directory + "/" + dataset['ID'] + "/figures/" + str(i) + "_" + dataset['ID'] + "_window_eseman_kdt_1_gantt.png"
    #                 comp_png = base_directory + "/" + hrs + dataset['ID'] + "/figures/" + str(i) + "_" + dataset['ID'] + "_window_eseman_kdt_" + hr + "_gantt.png"
    #                 cssim += compare_images(base_png, comp_png)
    #             ssim_list[dataset['name']].append(cssim / 10)
    #         except:
    #             ssim_list[dataset['name']].append(np.nan)
    #             fetch_list[dataset['name']].append(np.nan)
    # with open('ssim_fetch_lists.json', 'w') as f:
    #     json.dump({'ssim_list': ssim_list, 'fetch_list': fetch_list}, f, indent=2)
    with open('ssim_fetch_lists.json', 'r') as f:
        data = json.load(f)
        ssim_list = data['ssim_list']
        fetch_list = data['fetch_list']

    # Create a figure with two subplots: main plot and SSIM dot size legend

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0])

    # Prepare data for each dataset (excluding DGEMM, K-Means, LULESH)
    plot_datasets = [d for d in datasets if d['name'] not in ['DGEMM', 'K-Means', 'LULESH']]
    colors_fetch = [color[0], color[6], color[7]]
    colors_ssim = [color[0], color[6], color[7]]

    for idx, dataset in enumerate(plot_datasets):
        fetch = fetch_list[dataset['name']]
        ssim_vals = ssim_list[dataset['name']]
        # Plot fetch_list as a line
        ax.plot(hrds, fetch, marker='s', label=f"{dataset['name']} Fetch Time", color=colors_fetch[idx], linestyle='--', linewidth=3)
        # Plot SSIM as dots with radius proportional to (1-SSIM)
        min_r, max_r = 30, 300
        ssim_inv = [1 - s if s is not None and not np.isnan(s) else 0 for s in ssim_vals]
        if np.nanmax(ssim_inv) > 0:
            radii = [min_r + (max_r - min_r) * (v / np.nanmax(ssim_inv)) if np.nanmax(ssim_inv) > 0 else min_r for v in ssim_inv]
        else:
            radii = [min_r for _ in ssim_inv]
        ax.scatter(hrds, fetch, s=radii, color=colors_ssim[idx], alpha=1, label=f"{dataset['name']} 1-SSIM (dot size)", edgecolor='black', zorder=10)

    ax.set_ylabel('Average Fetch Time (ms)', fontsize=24)
    ax.set_xlabel('Pixel Window', fontsize=24)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.yaxis.set_major_formatter(lambda x, p: f'{x/1000:.0f}')
    ax.grid(True)

    # Interactivity threshold line
    ax.axhline(y=100000, color='black', linestyle='--', label='Interactivity Threshold')
    ax.text(
        1.7, 
        ax.get_ylim()[1] if ax.get_ylim()[1] < 120000 else 101000,
        'Interactivity Threshold',
        fontsize=18,
        color='black',
        ha='center',
        va='bottom',
    )

    # Legend
    fetch_handles = []
    ssim_handles = []
    for idx, dataset in enumerate(plot_datasets):
        fetch_handles.append(Line2D([0], [0], color=colors_fetch[idx], linestyle='--', marker='s', linewidth=3, label=f"{dataset['name']} Fetch Time"))
        ssim_handles.append(Line2D([0], [0], color=colors_ssim[idx], marker='o', linestyle='', markerfacecolor=colors_ssim[idx], markeredgecolor='black', markersize=10, label=f"{dataset['name']} 1-SSIM (dot size)"))
    legend_handles = []
    legend_labels = []
    for f in fetch_handles:
        legend_handles.extend([f])
        legend_labels.extend([f.get_label()])
    for f in ssim_handles:
        legend_handles.extend([f])
        legend_labels.extend([f.get_label()])
    ax.legend(legend_handles, legend_labels, fontsize=18, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.23), frameon=False, columnspacing=1.5)

    # SSIM dot size legend as a subplot
    min_ssim = min([min([1 - s for s in ssim_list[d['name']] if s is not None and not np.isnan(s)]) for d in plot_datasets])
    max_ssim = max([max([1 - s for s in ssim_list[d['name']] if s is not None and not np.isnan(s)]) for d in plot_datasets])
    legend_vals = np.linspace(min_ssim, max_ssim, 7)
    min_r, max_r = 30, 300
    if max_ssim > 0:
        radii = [min_r + (max_r - min_r) * (v / max_ssim) for v in legend_vals]
    else:
        radii = [min_r for _ in legend_vals]

    ax_legend = fig.add_subplot(gs[1])
    x = np.arange(len(legend_vals))
    ax_legend.scatter(x, np.ones_like(x), s=radii, color='gray', edgecolor='black', zorder=10)
    for i, v in enumerate(legend_vals):
        ax_legend.text(x[i], 1.15, f"{v:.2f}", ha='center', va='bottom', fontsize=20)
    ax_legend.set_xlim(-0.5, len(legend_vals) - 0.5)
    ax_legend.set_ylim(0.8, 1.4)
    ax_legend.axis('off')
    # Move the legend title below the chart
    ax_legend.text(
        0.5, 0, "1 - SSIM (dot size legend)",
        ha='center', va='top', fontsize=24, transform=ax_legend.transAxes
    )

    plt.tight_layout()
    print("Saving SSIM and Fetch Comparison Chart")
    plt.savefig('ssim_fetch_combined.png')
    plt.close()


def drawSSIMPlot(metric_type, alg_name):
    query_name = 'window'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize a dictionary to store data for each algorithm
    dataset_names = []
    dataset = datasets[1]
    hrds = ['1', '2', '4', '8', '16', '32', '64']
    alg_data = []

    for i, hr in enumerate(hrds):
        dataset_names.append(dataset['name'])
        if alg_name == 'eseman_kdt' and metric_type == 'data_fetch':
            metric_column = 'ds query time (micros)'
        elif metric_type == 'data_fetch':
            metric_column = 'Post Processing Time(micros)'
        elif metric_type == 'inp':
            metric_column = ' inp (micros)'
        elif metric_type == 'total_drawing':
            metric_column = 'total drawing time (micros)'
        hrd_text = ''
        if hr != '1':
            hrd_text = f"{hr}_"
        try:
            file_path = ""
            df = pd.DataFrame()
            if metric_type == 'data_fetch':
                file_path = base_directory + "/" + hrd_text + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"
                df = pd.read_csv(file_path, skipfooter=3, engine='python')
            elif metric_type == 'inp' or metric_type == 'total_drawing':
                file_path = base_directory + "/" + hrd_text +  dataset['ID'] + "/" + alg_name + "_" + query_name + "_selenium_check"
                df = pd.read_csv(file_path, skiprows=1)
            
            df = df.tail(10)
            
            if metric_column in df.columns:
                alg_data.append(df[metric_column].mean())
            else:
                alg_data.append(np.nan)
        except:
            alg_data.append(np.nan)

    # Plot a line for each algorithm
    line_styles = [':', '-', '-', '-.', ':', '--', '-']  # Define different line styles
    markers = ['o', 's', '*', 'v', 'D', '^', '^']  # Define different markers
    
    ax.plot(hrds, alg_data, marker=markers[0], label=alg_name, linestyle=line_styles[0], color='black')

    ax.plot(hrds, [1.000, 0.9835, 0.9508, 0.9068, 0.8649, 0.8451, 0.8373], marker=markers[0], label='SSIM', linestyle=line_styles[0], color='black')

    # # Set colors using the 'tab10' color palette
    # colors = plt.cm.tab10(np.linspace(0, 1, len(alg)))
    # for i, alg_name in enumerate(alg):
    #     ax.get_lines()[i].set_color(colors[i])
    # ax.axhline(y=100000, color='g', linestyle='--', label='Interactivity Threshold')
    # ax.fill_between(hrds, 0, 100000, color='g', alpha=0.1)

    ax.set_xlabel("Datasets")
    if metric_type == 'data_fetch':
        ax.set_title(f"Data Fetch Time Comparison - {qn[query_name]}")
        ax.set_ylabel("Average Data Fetch Time (s)")
    elif metric_type == 'inp':
        ax.set_title(f"Interaction to Next Point Comparison - {qn[query_name]}")
        ax.set_ylabel("Average INP Time (s)")
    elif metric_type == 'total_drawing':
        ax.set_title(f"Total Drawing Time Comparison - {qn[query_name]}")
        ax.set_ylabel("Average Drawing Time (s)")
    ax.grid(True)
    ax.legend()

    # handles, labels = ax.get_legend_handles_labels()
    # sorted_indices = np.argsort(labels)
    ax.legend(hrds, fontsize=16)

    # plt.xticks(rotation=45)
    # ax.set_yscale('log')
    ax.yaxis.set_major_formatter(lambda x, p: f'{x/1000000:.1f}')

    # Increase font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.set_title(ax.get_title(), fontsize=16)

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{query_name}_{metric_type}_comparison.png")
    plt.close()

def drawMemoryPlot(query_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize a dictionary to store data for each algorithm
    alg_data = {a: [] for a in alg}
    dataset_names = []

    # Collect data for each dataset and algorithm
    for i, dataset in enumerate(datasets):
        dataset_names.append(dataset['name'])
        for alg_name in alg:

            if not checkIfResultsGenerated(dataset, query_name, alg_name):
                alg_data[alg_name].append(np.nan)
                continue
            try:
                file_path = ""
                df = pd.DataFrame()
                
                file_path = base_directory + "/" + dataset['ID'] + "/" + alg_name + "_" + query_name + "_merged.csv"

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    second_last_line = lines[-2]
                    last_line = lines[-1]
                memory_value = float(second_last_line.split(':')[1].strip().split()[0])
                if alg_name == 'eseman_kdt' or alg_name == 'eseman_kdt_twod':
                    memory_value += float(last_line.split(':')[1].strip().split()[0])
                alg_data[alg_name].append(memory_value/1000)
            except:
                alg_data[alg_name].append(np.nan)

    # # Plot a line for each algorithm
    # for alg_name in alg:
    #     ax.plot(dataset_names, alg_data[alg_name], marker='o', label=alg_name)

    line_styles = [':', '-', '-', '-.', ':', '--']  # Define different line styles
    markers = ['o', 's', '*', 'v', 'D', '^']  # Define different markers
    # for i, alg_name in enumerate(alg):
    #     ax.plot(dataset_names, alg_data[alg_name], marker=markers[i], label=alg_code[i], linestyle=line_styles[i], color='black')
    hatch_patterns = ['/', '\\', '|', '-', 'o', 'x']

    bar_width = 0.12
    group_gap = 1.25
    x = np.arange(len(dataset_names))
    for i, alg_name in enumerate(alg):
        hatch = hatch_patterns[i % len(hatch_patterns)]
        bar_alpha = 0.5
        ax.bar(
            x * group_gap + i * bar_width,
            alg_data[alg_name],
            width=bar_width,
            label=alg_code[i],
            color=color[i % len(color)],
            hatch=hatch,
            edgecolor='black',
            alpha=bar_alpha,
            linewidth=1.5
        )

    ax.set_xticks(x * group_gap + bar_width * (len(alg) - 1) / 2)
    ax.set_xticklabels(dataset_names, fontsize=20)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Total memory consumption (GB)")
    ax.set_title(f"Memory Consumption Comparison")
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(fontsize=20)
    # handles, labels = ax.get_legend_handles_labels()
    # sorted_indices = np.argsort(labels)
    # ax.legend([handles[i] for i in sorted_indices], [labels[i] for i in sorted_indices], fontsize=16)

    # Increase font sizes
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.set_title(ax.get_title(), fontsize=20)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{query_name}_memory_comparison.png")
    plt.close()

def plotAllDatasets():
    for dataset in datasets:
        for q in query:
            if not q.endswith('picture'):  # Skip picture queries
                drawPlot(dataset['ID'], q)

def drawConstructionPlot():
    df = pd.read_csv('construction_time.csv')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique datasets and algorithms
    datasets = df['dataset'].unique()
    algorithms = df['algorithm'].unique()
    
    # Set width of bars and positions
    width = 0.8 / len(datasets)
    x = np.arange(len(algorithms))
    
    # Plot bars for each dataset
    for i, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset]
        offset = (i - len(datasets)/2 + 0.5) * width
        ax.bar(x + offset, data['time(seconds)'], width, label=dataset)
    
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Construction Time (seconds)')
    ax.set_title('Construction Time Comparison by Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('construction_time_comparison.png')
    plt.close()

if __name__ == '__main__':
    # generateTimeDistribution('kmeans', 'window')
    # resultChecker()

    # drawPlot(query[0], 'data_fetch', False)
    # drawPlot(query[1], 'data_fetch', False)

    # drawPlot(query[0], 'ssim', False)
    # # # drawPlot(query[0], 'inp')
    # drawPlot(query[0], 'total_drawing', True)
    # drawPlot(query[1], 'total_drawing', False)
    # drawPlot(query[0], 'inp', False)

    # dataTester(query[0], 'data_fetch', False)
    # dataTester(query[1], 'total_drawing', False)

    # drawingFetchingComp(query[0])
    # totalDrawingFetchingPercentage(query[0])
    
    # # # drawPlot(query[1], 'inp')
    # drawPlot(query[1], 'total_drawing', False)
    
    # drawMemoryPlot('window')
    # drawConstructionPlot()

    # drawSSIMPlot('data_fetch', 'eseman_kdt')
    # dgemm construction time data
    # cv = [39.4, 26.2, 64.2, 103.2, 83.2, 178.5, 170.4, 145.9, 158.5]
    # min_val = min(cv)
    # cv_normalized = [x - min_val for x in cv]
    # cv_cumsum = np.cumsum(cv_normalized)
    # print(cv_cumsum)

    # drawSSIMSmallMultiple()

