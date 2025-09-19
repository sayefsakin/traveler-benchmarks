import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def draw_sism_chart(dataset_name):
    csv_file = 'ssim_' + dataset_name +'.csv'
    # Load data from CSV file
    data = pd.read_csv(csv_file)
    print("Drawing line chart from CSV file:", csv_file)
    # Plot the line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='window', y='agc', data=data, label='Agglomerative Clustering')
    sns.lineplot(x='window', y='eseman', data=data, label='Eseman KDT with Midpoint Splitting')
    plt.title(dataset_name + ' SSIM' + '(Horizontal Resolution = 4236)')
    plt.xlabel('Horizontal Resolution Divisor')
    x_values = data['window'].unique()
    plt.xscale('log')
    plt.xticks(x_values, x_values)
    plt.ylabel('SSIM')
    plt.grid(True)
    # plt.show()
    # print("Line chart drawn successfully.")
    plt.savefig(dataset_name + '_ssim_chart.png', dpi=300, bbox_inches='tight')

def draw_sism_chart_between_dataset():
    # Load data from CSV file
    dgemm_data = pd.read_csv('ssim_dgemm.csv')
    kmeans_data = pd.read_csv('ssim_kmeans.csv')
    # print("Drawing line chart from CSV file:", csv_file)
    # Plot the line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='window', y='agc', data=kmeans_data, label='kmeans')
    sns.lineplot(x='window', y='agc', data=dgemm_data, label='dgemm')
    plt.title('Agglomerative Clustering SSIM' + '(Horizontal Resolution = 4236)')
    plt.xlabel('Horizontal Resolution Divisor')
    x_values = dgemm_data['window'].unique()
    plt.xscale('log')
    plt.xticks(x_values, x_values)
    plt.ylabel('SSIM')
    plt.grid(True)
    # plt.show()
    # print("Line chart drawn successfully.")
    plt.savefig('agc_ssim_chart.png', dpi=300, bbox_inches='tight')

def calc_query_time():
    datasets = ['dgemm', 'kmeans']
    algorithms = ['eseman_kdt_md', 'agglomerative_clustering']
    lines = {'eseman_kdt_md':'-', 'agglomerative_clustering':':'}
    labels = {'eseman_kdt_md':'ESEMAN KDT with Max Distance Splitting', 'agglomerative_clustering':'Agglomerative Clustering'}
    bins = [1, 2, 4, 8, 16, 32, 1024]
    
    # Create the line plot
    plt.figure(figsize=(10, 6))
    for algorithm in algorithms:
        plot_data = {'bins': [], 'dgemm': [], 'kmeans': []}
        for bin in bins:
            plot_data['bins'].append(bin)
        for dataset in datasets:
            for bin in bins:
                flie_name = './' + algorithm + '/' + dataset + '/load_time_bin' + str(bin) + '.txt'
                data = pd.read_csv(flie_name)
                # Get even numbered indices
                even_indices = range(1, len(data), 2)
                # Calculate average of query times at even indices
                even_avg = data.iloc[even_indices]['ds query time (micros)'].mean()
                # print(f"Bin {bin}: Average query time for even rows = {even_avg:.4f}")

                if dataset == 'dgemm':
                    plot_data['dgemm'].append(even_avg)
                else:
                    plot_data['kmeans'].append(even_avg)
    # print("Plot data:", plot_data)
    
        plt.plot(plot_data['bins'], plot_data['dgemm'], linestyle=lines[algorithm], label='DGEMM ' + labels[algorithm])
        plt.plot(plot_data['bins'], plot_data['kmeans'], linestyle=lines[algorithm], label='KMeans ' + labels[algorithm])
    plt.xlabel('Horizontal Resolution Divisor') 
    plt.ylabel('Average Query Time (microseconds)')
    plt.title('Average Query Time (Horizontal Resolution = 4236)')
    # ESEMAN KDT with Max Distance Splitting
    plt.xscale('log')
    plt.xticks(plot_data['bins'], plot_data['bins'])
    plt.legend()
    plt.savefig('query_time_comparison_line.png', dpi=300, bbox_inches='tight')
    plt.close()

def calc_avg_load_memory():
    datasets = ['dgemm', 'kmeans']
    algorithm = 'eseman_kdt_md'
    bins = [1, 2, 4, 8, 16, 32, 1024]
    flie_name = 'runtime_loadtime_memory.csv'
    data = pd.read_csv(flie_name)
    
    plot_data = {'dataset': [], 'kdt_load_time': [], 'agc_load_time': [], 'kdt_memory': [], 'agc_memory': []}
    for dataset in datasets:
        f_data = data[data['dataset'] == dataset]
        plot_data['dataset'].append(dataset)
        plot_data['kdt_load_time'].append(f_data['kdt_load_time(micros)'].mean())
        plot_data['agc_load_time'].append(f_data['agc_load_time(micros)'].mean())
        plot_data['kdt_memory'].append(f_data['kdt_memory(MB)'].mean())
        plot_data['agc_memory'].append(f_data['agc_memory(MB)'].mean())
    #     print(f_data)
    # print("Plot data:", plot_data)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot load time comparison
    x = range(len(plot_data['dataset']))
    width = 0.35

    ax1.bar([i - width/2 for i in x], plot_data['kdt_load_time'], width, label='ESEMAN KDT with Max Distance Splitting')
    ax1.bar([i + width/2 for i in x], plot_data['agc_load_time'], width, label='Agglomerative Clustering')
    ax1.set_yscale('log')
    ax1.set_ylabel('Load Time (microseconds)')
    ax1.set_title('Load Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_data['dataset'])
    ax1.legend()

    # Plot memory usage comparison 
    ax2.bar([i - width/2 for i in x], plot_data['kdt_memory'], width, label='ESEMAN KDT with Max Distance Splitting')
    ax2.bar([i + width/2 for i in x], plot_data['agc_memory'], width, label='Agglomerative Clustering')
    ax2.set_yscale('log')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_data['dataset'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('load_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # DGEM_ID = 'a9bd20ca-c4f2-4b54-8c49-b968ae7e78be'#'589ca754-ef75-426c-8d51-841cc61dc84a'
    # KMEANS_ID = '8b3289c9-a740-4091-a56d-e4d55af526b5'#'faf17535-2f66-4621-995f-49c7dbd84e8b'
    # LULESH_ID = '772c7330-d4eb-485b-866a-3b315063f9af'

    # if len(sys.argv) < 3:
    #     print("Usage: python calc_dssim.py <file1> <file2>")
    #     return
    dataset_name = 'dgemm'# 'kmeans'

    # draw_sism_chart(dataset_name)
    # draw_sism_chart_between_dataset()
    # calc_query_time()
    calc_avg_load_memory()

if __name__ == '__main__':
    main()