from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import io
import base64

# Generate sample data for demonstration
def generate_sample_data():
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                      random_state=42, cluster_std=1.5)
    return pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

app_ui = ui.page_fluid(
    ui.h1("Simple Clustering App"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Data Input"),
            ui.input_file("file", "Upload CSV file (optional)", 
                         accept=[".csv"], multiple=False),
            ui.input_action_button("use_sample", "Use Sample Data", 
                                 class_="btn-primary"),
            ui.br(), ui.br(),
            
            ui.h3("Clustering Parameters"),
            ui.input_numeric("n_clusters", "Number of Clusters", 
                           value=3, min=2, max=10),
            ui.input_checkbox("scale_data", "Scale Data", value=True),
            ui.input_action_button("run_clustering", "Run Clustering", 
                                 class_="btn-success"),
            ui.br(), ui.br(),
            
            ui.h3("Data Preview"),
            ui.output_text("data_info"),
            width=3
        ),
        ui.h3("Results"),
        ui.output_plot("cluster_plot"),
        ui.br(),
        ui.h4("Cluster Centers"),
        ui.output_table("cluster_centers"),
        ui.br(),
        ui.h4("Data with Cluster Labels"),
        ui.output_table("clustered_data")
    )
)

def server(input, output, session):
    # Reactive value to store the current dataset
    data = reactive.Value(None)
    clustered_result = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.use_sample)
    def load_sample_data():
        sample_data = generate_sample_data()
        data.set(sample_data)
    
    @reactive.Effect
    @reactive.event(input.file)
    def load_uploaded_data():
        if input.file() is not None:
            try:
                df = pd.read_csv(input.file()[0]["datapath"])
                # Only keep numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    data.set(df[numeric_cols])
                else:
                    data.set(None)
            except Exception as e:
                data.set(None)
    
    @reactive.Effect
    @reactive.event(input.run_clustering)
    def perform_clustering():
        if data.get() is not None:
            df = data.get().copy()
            
            # Use first two numeric columns for clustering
            feature_cols = df.columns[:2]
            X = df[feature_cols].values
            
            # Scale data if requested
            if input.scale_data():
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=input.n_clusters(), 
                          random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Store results
            result = {
                'data': df.copy(),
                'features': X,
                'labels': cluster_labels,
                'centers': kmeans.cluster_centers_,
                'feature_names': feature_cols.tolist(),
                'scaled': input.scale_data()
            }
            clustered_result.set(result)
    
    @output
    @render.text
    def data_info():
        if data.get() is not None:
            df = data.get()
            return f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns"
        else:
            return "No data loaded. Upload a CSV file or use sample data."
    
    @output
    @render.plot
    def cluster_plot():
        if clustered_result.get() is not None:
            result = clustered_result.get()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data points colored by cluster
            scatter = ax.scatter(result['features'][:, 0], 
                               result['features'][:, 1],
                               c=result['labels'], 
                               cmap='viridis', 
                               alpha=0.7, 
                               s=50)
            
            # Plot cluster centers
            ax.scatter(result['centers'][:, 0], 
                      result['centers'][:, 1],
                      c='red', 
                      marker='x', 
                      s=200, 
                      linewidths=3,
                      label='Centroids')
            
            ax.set_xlabel(f"{result['feature_names'][0]}" + 
                         (" (scaled)" if result['scaled'] else ""))
            ax.set_ylabel(f"{result['feature_names'][1]}" + 
                         (" (scaled)" if result['scaled'] else ""))
            ax.set_title(f"K-Means Clustering Results (k={input.n_clusters()})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            plt.tight_layout()
            return fig
        else:
            # Show empty plot with instructions
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Load data and run clustering to see results', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14)
            ax.set_title('Clustering Visualization')
            return fig
    
    @output
    @render.table
    def cluster_centers():
        if clustered_result.get() is not None:
            result = clustered_result.get()
            centers_df = pd.DataFrame(
                result['centers'],
                columns=[f"{name}" + (" (scaled)" if result['scaled'] else "") 
                        for name in result['feature_names']]
            )
            centers_df.index = [f"Cluster {i}" for i in range(len(centers_df))]
            return centers_df.round(3)
        else:
            return pd.DataFrame()
    
    @output
    @render.table
    def clustered_data():
        if clustered_result.get() is not None:
            result = clustered_result.get()
            df_with_clusters = result['data'].copy()
            df_with_clusters['Cluster'] = result['labels']
            # Show only first 100 rows to avoid overwhelming the display
            return df_with_clusters.head(100)
        else:
            return pd.DataFrame()

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)
