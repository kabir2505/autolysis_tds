# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "scikit-learn",
#   "networkx",
#   "requests",
#   "chardet",
# "Pillow",
# ]
# ///

import os,sys,json,glob
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import seaborn as sns
import chardet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import base64
from PIL import Image
import re

#check if the token is set
if "AIPROXY_TOKEN" not in os.environ:
    print("AIPROXY_TOKEN not set")
    exit(1)

api_key=os.environ["AIPROXY_TOKEN"] #api key for aiproxy


#validate command line arguments #uv run autolysis.py dataset.csv
if len(sys.argv) != 2:
    print("Usage: autolysis.py <dataset.csv>")
    exit(1)
    
csv_file=sys.argv[1]
file_name_without_extension = csv_file.split('.csv')[0]
#file existence
if not Path(csv_file).is_file():
    print(f"File {csv_file} does not exist")
    exit(1)


try:
    with open(csv_file, "rb") as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
    data=pd.read_csv(csv_file,encoding=encoding)
    print("dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)
    
    
file_name_without_extension =os.path.splitext(os.path.basename(csv_file))[0]

# output_dir = f"./{file_name_without_extension}"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
    

    

#generic analysis
def generic_analysis(which):
    #data = data.where(pd.notnull(data), None)
    """
    Perform a generic analysis of the dataset.

    Parameters
    ----------
    which : str
        Either "story" or "context". If "story", the analysis will include
        additional metadata for storytelling purposes.

    Returns
    -------
    dict
        A dictionary containing the results of the analysis.
    """
    n_samples,n_features=data.shape # number of samples and features
  


    # Build the return object
    return_object = {
        "file_name": csv_file,
        "n_samples": int(len(data)),
        "n_features": int(data.shape[1]),
        "feature_info": {
            col: {
                "type": data[col].dtype.name,
                "missing": int(data[col].isna().sum()),
                "n_unique": int(data[col].nunique()),
                **(
                    {
                        "mean": float(data[col].mean()),
                        "std": float(data[col].std()),
                        "min": float(data[col].min()),
                        "max": float(data[col].max()),
                        "range": float(data[col].max() - data[col].min()),
                        "median": float(data[col].median()),
                        "skew": float(data[col].skew()),
                        "5_unique": data[col].unique()[:5].tolist() if which=="story" else [],
                        "percentiles": data[col].quantile(
                            [0.01, 0.05, 0.25, 0.50, 0.75,0.99]
                        ).to_dict(),
                        "5_value_counts": data[col].value_counts()
                        .head(5).to_dict() if which=="story" else {},
                        "mode_proportion": float(data[col].value_counts().max() / len(data[col])),
                        "memory_usage": int(data[col].memory_usage(deep=True)),
                    }
                    if data[col].dtype.name in ["float64", "int64"]
                    else {}
                ),
                **(
                    {
                        "unique": data[col].unique()[:5].tolist(),
                        "5_value_counts": data[col].value_counts() 
                        .head(5)
                        .to_dict() if which=="story" else {},
                        "top": data[col].value_counts().idxmax(),
                        "freq": int(data[col].value_counts().max()),
                    }
                    if data[col].dtype.name == "object"
                    else {}
                ),
            }
            for col in data.columns
        },
      
    }

    return return_object
   


# print(json.dumps(generic_analysis(which="context")))


#-------------------SURE-------------------------
def plot_missing_data():
    """
    Plot a bar chart of the proportion of missing data for each feature in the dataset.

    This function calculates the proportion of missing values for each column in the 
    dataset and visualizes it using a bar chart. The percentage of missing data is 
    annotated on each bar for better readability. The plot is saved as an image file.

    Parameters:
    None

    Returns:
    None
    """
    df = data.copy()
    missing_distri = (df.isna().sum() / df.shape[0]).sort_values(ascending=False)

    # Create the bar plot with color palette
    ax = missing_distri.plot.bar(color='skyblue', edgecolor='black', figsize=(12, 6))
    plt.title('Missing Data Distribution', fontsize=16, weight='bold')
    
    # Total count for percentage calculation
    total_count = missing_distri.sum()

    # Annotate each bar with the percentage of missing data
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total_count) * 100
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2, height), 
                    ha='center', va='bottom', fontsize=10, weight='bold')

    # Set axis labels with larger font size and bold
    plt.xlabel('Features', fontsize=12, weight='bold')
    plt.ylabel('Proportion of Missing Data', fontsize=12, weight='bold')
    
    # Rotate x-axis labels for better readability if necessary
    plt.xticks(rotation=45, ha='right')

    # Save the plot with adjusted layout
    plt.tight_layout()
    plt.savefig(f"{file_name_without_extension}_missing_data_distri.png", dpi=100, bbox_inches="tight")
def detect_outliers(columns):
    """
    Identify outliers in a given set of columns.

    Parameters
    ----------
 
    columns : list
        The names of the columns to check for outliers.

    Returns
    -------
    anomalies : dict
       dict: Anomalies for each numeric column
    """
    z_threshold=3
    anomalies = {}
    for col in columns:
        z_score = np.abs(stats.zscore(data[col]))
        col_anomalies = data[z_score > z_threshold]
        if not col_anomalies.empty:
            anomalies[col]={
                    'total_anomalies': len(col_anomalies),
                    'anomaly_percentage': len(col_anomalies) / len(data) * 100,
                    'anomaly_details': col_anomalies[col].tolist()
                }
    return anomalies
        

def plot_correlation():
    """
    Plot and save the correlation matrix for numeric columns in the dataset.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    filename (str): The filename to save the plot. Defaults to 'correlation_matrix.png'.
    
    Returns:
    None
    """
    # Compute the correlation matrix for numeric columns
    corr_matrix = data.select_dtypes(include=["float64", "int64"]).corr()
    
    # Plot the heatmap
    plt.figure(figsize=(4, 4))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 6})
    
    plt.title("Correlation Matrix", fontsize=10)
    
    # Save the plot to a file with low resolution
    plt.savefig(f"{file_name_without_extension}_correlation_matrix.png", dpi=100, bbox_inches="tight")  # Lower dpi for smaller file size

def perform_clustering_for_story(n_clusters):
    """
    Perform KMeans clustering and return metadata for storytelling.

    Parameters:
    data (pd.DataFrame): The dataset to cluster.
    n_clusters (int): The number of clusters.

    Returns:
    dict: Metadata about the clustering results for storytelling.
    """
    df=data.copy()
    # Separate features and preprocess
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_features = df.select_dtypes(include=["object", "category"]).columns

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Clustering pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))])

    # Handle missing values
    data_cleaned = df.dropna()

    # Fit and predict
    cluster_labels = pipeline.fit_predict(data_cleaned)

    # Collect cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index().to_dict()

    # Extract numeric summaries for each cluster
    cluster_data = data_cleaned.copy()
    cluster_data["Cluster"] = cluster_labels
    cluster_summaries = (
        cluster_data.groupby("Cluster")
        .agg({col: ["mean", "min", "max"] for col in numeric_features})
        .to_dict()
    )

    # Return metadata for storytelling
    metadata = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "cluster_summaries": cluster_summaries,
        "categorical_features_used": list(categorical_features),
        "numeric_features_used": list(numeric_features),
    }

    return metadata


def plot_categorical_columns(max_categories=5, max_unique=230):
    """
    Draw countplots for categorical columns, limiting the number of categories to the top N most frequent,
    and skipping columns with more than a specified number of unique categories.
    
    :param max_categories: Maximum number of categories to plot (only top N frequent categories).
    :param max_unique: Maximum number of unique categories to consider for plotting.
    """
    # Select categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Filter out columns with more than max_unique unique categories
    valid_cols = [col for col in categorical_cols if data[col].nunique() <= max_unique]
  
    # If no valid columns exist, print a message and return
    if not valid_cols:
        print(f"No categorical columns with {max_unique} or fewer unique categories.")
        return

    if len(valid_cols) > 3:
        valid_cols=valid_cols[:3]
    
    # Determine the number of rows and columns needed for subplots
    max_columns_per_row = 3
    ncols = min(len(valid_cols), max_columns_per_row)  # Ensure no more than max_columns_per_row
    nrows = (len(valid_cols) + ncols - 1) // ncols  # This ensures that we fit all columns
    
    # Create subplots with the calculated number of rows and columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]  
    else:
        axes = axes.flatten()  
    
    # Loop through the valid categorical columns and plot countplots
    for i, col in enumerate(valid_cols):
        # Count the number of unique categories
       
        num_categories = data[col].nunique()
        
        # If there are more than max_categories, limit to top N frequent categories
        if num_categories > max_categories:
            # Get the top N most frequent categories
            top_categories = data[col].value_counts().head(max_categories).index
            data_to_plot = data[data[col].isin(top_categories)]
        else:
            # Otherwise, use the entire column
            data_to_plot = data
        
        # Plot countplot for the selected data
        sns.countplot(x=col, data=data_to_plot, ax=axes[i], palette='Set2')  # Added color palette for better contrast
        axes[i].set_title(f'Countplot of {col} ({num_categories} categories)', fontsize=12, weight='bold')
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        axes[i].set_xlabel(col, fontsize=10, weight='bold')  # Add xlabel
        axes[i].set_ylabel('Count', fontsize=10, weight='bold')  # Add ylabel
    
    # Turn off axes for any unused subplots
    for j in range(len(valid_cols), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{file_name_without_extension}_categorical_columns_countplots.png")

#----------------------------function calling----------------------
def plot_trends_over_time(time_column, value_column, plot_title="Trend Analysis"):
    """
    Plot trends of a specific column over time.
    
    :param data: DataFrame containing data.
    :param time_column: Column name for the time (e.g., publication year).
    :param value_column: Column name for the metric (e.g., average rating).
    :param plot_title: Title for the plot.
    :return: None
    """
    df = data
    trend_data = df.groupby(time_column)[value_column].mean()
    
    # Plot the trend
    plt.figure(figsize=(12, 6))
    trend_data.plot(kind='line', color='teal', linewidth=2, linestyle='-', marker='o', markersize=6)
    
    # Enhance the plot with titles, labels, and grid
    plt.title(plot_title, fontsize=16, fontweight='bold')
    plt.xlabel(time_column, fontsize=12)
    plt.ylabel(value_column, fontsize=12)
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot to the specified directory
    plt.savefig(f"{file_name_without_extension}_trends_over_time.png")
   

def plot_histograms_for_numerical_columns(cols, ncols=3):
    """
    Plot histograms with KDE for specific columns in the DataFrame.

    :param cols: List of columns to plot.
    :param ncols: Number of columns for subplots.
    """
    if cols and len(cols)>3:
        cols=cols[:3]
    numerical_cols = [col for col in cols if data[col].dtype in ['int64', 'float64']]
    
    if not numerical_cols:
        print("No numerical columns found in the provided list.")
        return

    # Determine subplot grid dimensions
    nrows = (len(numerical_cols) + ncols - 1) // ncols

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    
    # Ensure `axes` is always a 1D array
    if nrows == 1:
        axes = np.array(axes).flatten()  # Single row
    elif ncols == 1:
        axes = np.array(axes).flatten()  # Single column
    else:
        axes = axes.flatten()

    # Plot histograms for each numerical column
    for i, col in enumerate(numerical_cols):
        sns.histplot(data[col], kde=True, ax=axes[i], color='skyblue', kde_kws={'color': 'red'})
        axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')

    # Turn off unused subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the plot
    plt.tight_layout()
   
    plt.savefig(f"{file_name_without_extension}_numerical_cols_hist.png")



def create_and_plot_network(source_col, target_col,):
    """
    Save network graphs as images.


    :param source_col: Source node name.
    :param target_col: Target node name.
    :param save_path: Path to save the visualization.
    :return: None
    """
    G = nx.from_pandas_edgelist(data, source=source_col, target=target_col)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    plt.title("Network Graph Analysis")
    plt.tight_layout()
    plt.savefig(f"{file_name_without_extension}_network_graph.png")
    plt.close()



#------
def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in function {func.__name__}: {e}")
        return None  

safe_run(plot_correlation)
safe_run(plot_categorical_columns)
safe_run(plot_missing_data)
anomalies_dict=safe_run(detect_outliers, data.select_dtypes(["float", "int"]).columns)
clustering_info=safe_run(perform_clustering_for_story, n_clusters=4)


#----------LLM Code-----------------------------------------------

function_descriptions = [
    {
        "name": "plot_trends_over_time",
        "description": "    Plot trends of a specific column over time.",
        "parameters": {
            "type": "object",  
            "properties": {
                "time_column": {  # Nested under 'properties'
                    "type": "string",
                    "description": "Name of the column containing time or date information."
                },
                "value_column": {
                "type": "string",
                "description": "Name of the column containing values to be plotted over time."
            }
            },
            "required": ["time_column""value_column"]  # Specify 'columns' as required
        }
    },
  {
    "name": "plot_histograms_for_numerical_columns",
    "description": "Plot histograms with KDE for specific numerical columns in a dataset.",
    "parameters": {
        "type": "object",
        "properties": {
            "cols": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of column names to plot histograms for."
            },
            "ncols": {
                "type": "integer",
                "default": 3,
                "description": "Number of columns in each row of subplots."
            }
        },
        "required": ["cols"]
    }
},
    {
    "name": "create_and_plot_network",
    "description": "Generate and save a network graph visualization based on source and target columns.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_col": {
                "type": "string",
                "description": "Column name representing the source nodes in the network."
            },
            "target_col": {
                "type": "string",
                "description": "Column name representing the target nodes in the network."
            }
        },
        "required": ["source_col", "target_col"]
    }
}
    
]















headers={"Authorization":f"Bearer{api_key}"}
model="gpt-4o-mini"

send_prompt = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": """You are a brilliant and detail-oriented data analyst with a strong understanding of various data analysis techniques. Your task is to analyze the provided dataset by selecting and invoking the most appropriate functions based on the user's request and the context provided.

            You must carefully examine the dataset context, which includes details like the types of data (numerical, categorical), the dataset size, missing values, and any specific analysis requirements. Then, choose the function(s) that are best suited to the analysis task at hand.

            You are expected to:
            - Select functions that align with the nature of the data (e.g., use statistical analysis for numerical data, visualizations for categorical data, etc.).
            - Ensure the parameters for each function are accurate and appropriate.
            - Do not return any explanations or any text, just select the appropriate functions and arguments for them based on the context is all
            -  if there is a date column or similar, call the time series analysis function etc
            - you are allowed to call more than one function
            """
        },
        {
            "role": "user",
            "content": f"""I need help performing a data analysis on the given dataset. The context for this analysis is as follows: {json.dumps(generic_analysis(which="context"))}. Based on the dataset characteristics, please choose the most relevant function(s) to perform the analysis and explain the results."""
        }
    ],
    "functions": function_descriptions,
}
r = requests.post(url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=send_prompt)
response_data=r.json()
# print('output',response_data
#       )

def parse_function_calls(response_data):
    """
    Parse function calls from the assistant's response and return them as a list of function names and arguments.
    
    Parameters:
    ----------
    response_data : dict
        The response data from the assistant which contains the function call suggestions.
        
    Returns:
    -------
    function_calls : list
        A list of dictionaries containing function names and their respective arguments.
    """
    function_calls = []
    
    try:
      
        for choice in response_data.get("choices", []):
        
            function_call = choice.get("message", {}).get("function_call", {})
            
            if function_call:
                function_name = function_call.get("name")
                arguments = function_call.get("arguments")
                
             
                if function_name and arguments:
                    try:
                        # Parse the arguments (if valid JSON string)
                        function_args = json.loads(arguments)
                        
                        # Append to the function calls list
                        function_calls.append({
                            "name": function_name,
                            "arguments": function_args
                        })
                    except json.JSONDecodeError as e:
                        # Log the error but don't stop the program
                        print(f"Error parsing arguments for function '{function_name}': {e}")
                else:
                    # Handle cases where either name or arguments are missing
                    print(f"Function call missing 'name' or 'arguments'. Skipping this function.")
            else:
                # Handle cases where 'function_call' is missing in the choice
                print("No valid 'function_call' found in this choice. Skipping.")
                
    except Exception as e:
        # Log any other unexpected error
        print(f"An unexpected error occurred while parsing function calls: {e}")
    
    return function_calls


function_calls = parse_function_calls(response_data)

function_map = {
    "plot_trends_over_time": plot_trends_over_time,
    "plot_histograms_for_numerical_columns": plot_histograms_for_numerical_columns,
    "create_and_plot_network": create_and_plot_network,
   
}

def execute_function_calls(function_calls):
    """
    Execute the function calls dynamically and handle any errors gracefully.
    
    Parameters:
    ----------
    function_calls : list
        List of dictionaries containing function names and their respective arguments.
        
    Returns:
    -------
    dict
        A dictionary with function names as keys and their respective results (or error messages) as values.
    """
    results = {}  # Store the results of function calls
    
    for function_call in function_calls:
        function_name = function_call.get("name")
        print('fn_name',function_name)
        arguments = function_call.get("arguments")
        print('args',arguments)
        if not function_name:
            results[function_name] = "Function name missing. Skipping this call."
            continue
        
        # Check if arguments are missing or invalid
        if not arguments:
            results[function_name] = f"Arguments missing for function '{function_name}'. Skipping this call."
            continue
        
        try:
            # Execute the function if it exists in the function_map
            if function_name in function_map:
                func = function_map[function_name]
                print("hello",func)
                # Execute the function with the unpacked arguments and capture the result
                result = func(**arguments)
                print("executed")
                
                # If function doesn't return anything, note that
                if result is None:
                    results[function_name] = f"Function '{function_name}' executed successfully but did not return anything."
                else:
                    results[function_name] = result
            else:
                results[function_name] = f"Function '{function_name}' is not recognized. Skipping."
        except Exception as e:
            results[function_name] = f"An error occurred while executing function '{function_name}': {e}"

    return results



execution_results = execute_function_calls(function_calls)
























#--------------------

def resize_image(image_path, size=(512, 512)):
    """
    Resizes an image to the specified size.
    :param image_path: Path to the image file.
    :param size: Tuple specifying the desired image size (width, height).
    :return: Path to the resized image.
    """
    resized_path = image_path.replace(".png", ".png")
    with Image.open(image_path) as img:
        img = img.resize(size, Image.LANCZOS)
        img.save(resized_path, format="PNG")
    return resized_path

def encode_image(image_path):
    """
    Encodes a given image file to a Base64 string.
    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Find all .png files in the output directory
png_files = glob.glob(f"*.png")

# Resize and encode all images
image_urls = []
for i,file in enumerate(png_files):
    resized_file = resize_image(file)  # Resize the image to 512x512
    base64_image = encode_image(resized_file)
    # Prepare the data:image/png;base64 URL
   
    image_url = f"data:image/png;base64,{base64_image}"
    image_urls.append({"type": "image_url", "image_url": {"url": image_url,"path":file}})
    break
    

# print(image_urls[0]['image_url']['path'],)


send_story_prompt={
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": """You are a Data Story Archaeologist, a 
            specialist in transforming complex data analysis into engaging 
            narratives. Your responses should acknowledge receipt of information 
            only, without generating any content until explicitly instructed.
            When instructed to generate, generate a MD narrative and only share that (no extra explanations), which will strictly include:
                The data context you received
                The analysis carried out
                The insights discovered
                The implications of findings (i.e. what to do with the insights)
            The MD writeup should be well-structured, using headers, lists, and emphasis appropriately"""
        },
        {
            "role": "user",
            "content": f"""
          Ready to receive the sacred data artifacts. Awaiting part 1 of 3.."
            """
        },
        {
            "role":"user",
            "content":f"""
            Behold! The Sacred Scroll of Statistics has been unearthed:

{generic_analysis(which="story")}

        Within this mystical object lies:
        - The Count of Samples (how many data points answered the call)
        - The Fellowship of Features (columns that band together)
        - The Types of Tales (data types that shape our story)
        - The Missing Mysteries (where data dares not tread)
        - The Unique Universe (distinct values that populate our world)
        - The Numerical Narratives (means, medians, and their magical kin)

        Please respond with: "Sacred Scroll of Statistics received and decoded. Awaiting part 2 .
            """
        },
        # {
        #     "role":"user",
        #     "content":f"""
            
        #     From the depths of the dataset dimension comes the Chronicles of Anomalies and K-means clustering information:

        #     {anomalies_dict} and {clustering_info}
        #     Please respond with: "Chronicles of Anomalies documented. Awaiting part 3 of 4."
        #     """
        # },
        
        {
            "role":"user",
            "content":[
                 {"type": "text", "text": f"""part 2: include an image and your interpreation of it, using the path mentioned here:{image_urls[0]['image_url']['path']}   Now that you possess all  sacred artifacts, it is time to weave your grand tale. Create a README.md that:
            
1. Opens with "The Legend of the Dataset" (executive summary)
2. Continues through "The Great Data Expedition" (methodology)
3. Reveals "The Statistical Secrets" (findings)
4. Shows "The Visual Prophecies" (plot interpretations)
5. Concludes with "The Future Insights" (recommendations)
    Very important: While including images in the mardown narrative, make sure to include their paths in the image_urls[image_url] field.

        Every number has a name, every outlier has an origin, and every correlation tells a tale. Make them sing in your markdown text and only pass this text!"""
        },
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": image_urls[0]['image_url']['url'],
                         "path": image_urls[0]['image_url']['path'],
                         "detail": "low"
                     }
                 }
            ]
        }
     
          
    ]
        }
        
    


r = requests.post(url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=send_story_prompt)

try:
    llm_response = r.json()  # Try to parse the JSON response
    # print('output2', llm_response)

    # Check if the expected keys are in the response
    if 'choices' not in llm_response or len(llm_response['choices']) == 0:
        raise ValueError("Invalid response structure: 'choices' not found in the response.")

    # Extract the content from the response
    content = llm_response['choices'][0].get('message', {}).get('content', None)
    
    # Check if content is present
    if not content:
        raise ValueError("No 'content' found in the response message.")
    
    # Prepare the output path for README.md
   
    match = re.search("```markdown\n(.*?)\n```", content, re.DOTALL)
    if match:
        content = match.group(1)
    # Writing to README.md
    with open("Readme.md", 'w') as f:
        f.write(content)

    print(f"README.md file has been written to Readme")

except json.JSONDecodeError:
    print("Error: Failed to decode JSON from the response.")
except KeyError as e:
    print(f"Error: Missing expected key in the response: {e}")
except ValueError as e:
    print(f"Error: {e}")
except OSError as e:
    print(f"Error: Failed to write to file. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")