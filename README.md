# Automated Analysis Script: autolysis.py

## Project Goal  
Develop a Python script leveraging LLMs to analyze, visualize, and narrate insights from any dataset provided as a CSV file. 

## Key Features  
1. **Generic Analysis**  
   - Performs summary statistics, missing value analysis, correlation matrices, clustering, and more.
2. **Interactive LLM Utilization**  
   - **Code Assistance**: Asks the LLM for analysis code and executes it.  
   - **Summary Generation**: Uses the LLM to narrate insights and create a compelling story.  
   - **Function Recommendations**: Leverages the LLM to identify insightful analysis steps.  
3. **Visualization**  
   - Generates PNG charts using libraries like Seaborn or Matplotlib.  
4. **Story Narration**  
   - Summarizes findings and implications into a Markdown file (`README.md`).  

Execute the script using uv:
uv run autolysis.py dataset.csv

### Outpus generated
•	README.md: Story-style summary of the analysis.
•	*.png: 1-3 data visualizations supporting the narrative.

Technologies Used
	•	Python
	•	LLM Integration: GPT-4o-Mini (via AI Proxy).
	•	Visualization: Seaborn, Matplotlib.

This project is licensed under the MIT License.
