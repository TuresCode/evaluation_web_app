import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import matplotlib.patches as patches
import zipfile
import plotly.graph_objects as go

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_visibility' not in st.session_state:
    st.session_state.analysis_visibility = False
if 'product_column' not in st.session_state:
    st.session_state.product_column = ""
if 'references' not in st.session_state:
    st.session_state.references = ""
if 'neg_control' not in st.session_state:
    st.session_state.neg_control = ""
if 'substrate_column' not in st.session_state:
    st.session_state.substrate_column = ""
if 'separator' not in st.session_state:
    st.session_state.separator = ""
if 'other_ref' not in st.session_state:
    st.session_state.other_ref = ""
if 'digits' not in st.session_state:
    st.session_state.digits = 2
if 'scientific_notation' not in st.session_state:
    st.session_state.scientific_notation = True

if 'compare_1' not in st.session_state:
    st.session_state.compare_1 = ""
if 'compare_2' not in st.session_state:
    st.session_state.compare_2 = ""
if 'select_area1' not in st.session_state:
    st.session_state.select_area1 = ""
if 'select_area2' not in st.session_state:
    st.session_state.select_area2 = ""
if 'use_plotly' not in st.session_state:
    st.session_state.use_plotly = True
if 'normalize_values' not in st.session_state:
    st.session_state.normalize_values = False
if 'use_conversion' not in st.session_state:
    st.session_state.use_conversion = False

if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None
if 'selectivity_result' not in st.session_state:
    st.session_state.selectivity_result = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Classic analysis"




def create_zip_file(files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, file_data in files.items():
            zip_file.writestr(file_name, file_data)
    return zip_buffer.getvalue()

def read_excel(file):
    # Read the Excel file
    df = pd.read_excel(file)
    #add new row at the top with df.columns
    new_row = pd.DataFrame([df.columns], columns=df.columns)
    df = pd.concat([new_row, df], ignore_index=True)
    # Reassign the column names to A,B,C etc
    df.columns = [f"{chr(65+i)}" for i in range(0, len(df.columns))]
    return df

def read_and_prepare_excel(file, separator=''):
    df = read_excel(file)

    #clean dataframe by deleting all rows which are not in the index
    if separator != '':
        df['A'] = df['A'].str.split(separator).str[1]

    #index_list contains only A1, A2 until H12
    index_list = [f"{chr(65+i)}{j}" for i in range(0, 8) for j in range(1, 13)]

    df = df[df['A'].isin(index_list)]
    #check if the shape is correct
    if df.shape[0] != 96:
        st.error("Couldn't find index A1, A2 or file does not have 96 rows. Please check the file and try again.")
        return None
    #drop duplicates if they exist based on index
    df = df.drop_duplicates()
    return df

def process_excel_file(file, product_column,separator='', substrate_column='',references='', use_conversion=False):

    df = read_and_prepare_excel(file, separator)

    # Get the data from the product column
    data = df[product_column]
    #fill NaN values with 0
    data = data.fillna(0)


    # If substrate column is provided, calculate the conversion
    if substrate_column != '' and use_conversion:
        st.info("Conversion is used to calculate FIOP.")
        substrate = df[substrate_column]
        substrate = substrate.fillna(0)
        new_data = data / (substrate + data) * 100
        data = new_data
        if references != '':
            st.info("References are used to calculate FIOP based on conversion.")
            references = references.split(',')
            ref_data = df[df['A'].isin(references)]
            ref_data = ref_data[product_column]    
            ref_data = ref_data.fillna(0)
            ref_conversion = ref_data / (df[df['A'].isin(references)][substrate_column] + ref_data) * 100
            mean_reference = ref_conversion.mean()   
            data = data / mean_reference * 100
    if references != '':
        st.info("References are used to calculate FIOP based on Area's.")
        references = references.split(',')
        ref_data = df[df['A'].isin(references)]
        ref_data = ref_data[product_column]    
        ref_data = ref_data.fillna(0)
        mean_reference = ref_data.mean()   
        data = data / mean_reference * 100
    return df, data

def compare_columns(file, col1, col2):
    df = read_and_prepare_excel(file, separator=st.session_state.separator)
    #get first column and col1 and col2
    comparison = df[['A', col1, col2]].copy()
    
    # Convert columns to numeric, replacing non-numeric values with NaN
    comparison[col1] = pd.to_numeric(comparison[col1], errors='coerce')
    comparison[col2] = pd.to_numeric(comparison[col2], errors='coerce')
    
    # Calculate difference and ratio only for rows where both values are numeric
    comparison['Difference'] = comparison[col1] - comparison[col2]
    comparison['Ratio'] = comparison[col1] / comparison[col2]
    
    return comparison

def calculate_selectivity(file, col1, col2):

    df = read_and_prepare_excel(file, separator=st.session_state.separator)
    #get first column and col1 and col2
    selectivity = df[['A', col1, col2]].copy()

    #convert columns to numeric, replacing non-numeric values with NaN
    selectivity[col1] = pd.to_numeric(selectivity[col1], errors='coerce')
    selectivity[col2] = pd.to_numeric(selectivity[col2], errors='coerce')
    
    selectivity['Selectivity'] = (selectivity[col1] - selectivity[col2]) / (selectivity[col1] + selectivity[col2])
    if 'sequence' in df.columns:
        selectivity['Sequence'] = df['sequence']
    return selectivity


def create_heatmap(data, references, neg_control, other_ref, digits, scientific_notation):
    fig, ax = plt.subplots(figsize=(14, 8))
    data = data.values.reshape(8, 12)
    data = pd.DataFrame(data, columns=range(1, 13), index=[chr(65+i) for i in range(0, 8)])
    
    # Create the heatmap
    if scientific_notation:
        digit_str = f".{digits}e"
    else:
        digit_str = f".{digits}f"
    ax = sns.heatmap(data, annot=True, square=True, linewidths=0.5, fmt=digit_str, cmap="Blues", ax=ax, cbar_kws={'shrink': .8})
    
    # Function to add rectangles
    def add_rectangle(well, color):
        row = ord(well[0]) - ord('A')
        col = int(well[1:]) - 1
        rect = patches.Rectangle((col, row), 1, 1, fill=False, edgecolor=color, lw=3)
        ax.add_patch(rect)
    
    # Add rectangles for references and negative controls
    if references != '':
        for ref in references.split(','):
            add_rectangle(ref.strip(), 'green')
    if neg_control != '':
        for neg in neg_control.split(','):
            add_rectangle(neg.strip(), 'red')
    if other_ref != '':
        for other in other_ref.split(','):
            add_rectangle(other.strip(), 'violet')


    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.1, top - 0.4)
    ax.set_xlabel("Column", labelpad=10)
    ax.set_ylabel("Row", labelpad=10)
    # Make the plot responsive
    
    return fig,data


st.set_page_config(
    page_title="96-Well Plate Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Sidebar
st.sidebar.title("96-Well Plate  Analyzer")
uploaded_file = st.sidebar.file_uploader("Select excel file", type="xlsx")
example = pd.read_excel('20240101_example.xlsx')
output = io.BytesIO()
example.to_excel(output, index=False)
output.seek(0)

# Create a download button in the sidebar
st.sidebar.download_button(
    label="Download example file",
    data=output,
    file_name="20240101_example.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
use_conversion = st.sidebar.checkbox("Use conversion", value=st.session_state.use_conversion, help="If checked, conversion will be used to calculate FIOP rather than areas of the product")

analysis_visibility = False

# Main content

st.title("96-Well Plate Analyzer")
if st.button("Pre-analysis"):
    if uploaded_file is not None:
        st.session_state.df = read_excel(uploaded_file)
        st.session_state.analysis_visibility = True
    else:
        st.warning("Please upload an Excel file.")

# Display preview if available
if st.session_state.analysis_visibility:
    st.subheader("Preview of the Excel file")
    #create arrow-compatible dataframe for preview
    st.dataframe(st.session_state.df.head().applymap(str))

# make distance to the tabs
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.analysis_visibility:
    st.stop()

tab1, tab2 = st.tabs(["Classic analysis", "Advanced analysis"])

with tab1:
    st.info("First column of the excel file (column A) is used as index. A1, A2, A3, ... H12. Select the product column eg. M and assign references and negative control if available.")
    col1, col2 = st.columns(2)
    
    with col1:
        product_column = st.text_input("Product Column [MANDATORY]", value=st.session_state.product_column, help="Choose the column that contains the product data. This is mandatory.")
        references = st.text_input("Assign references", value=st.session_state.references, help="If no reference are chosen, area will be used for plotting. Assign references by typing in well location, separate by using commas (,).")
        neg_control = st.text_input("Assign negative control", value=st.session_state.neg_control, help="Negative controls are only used for plotting. Assign references by typing in well location, separate by using commas (,).")
        digits = st.number_input("Digits", value=st.session_state.digits, help="Number of digits to round the values to.", min_value=0, max_value=4)
    with col2:
        substrate_column = st.text_input("Substrate Column", value=st.session_state.substrate_column, help="It's possible to perform analysis without choosing substrate column but if you want to use conversion, choose the substrate column and click the checkbox. Use conversion")
        separator = st.text_input("Separator in sample name", value=st.session_state.separator, help="If samples are labeled according to A1,A2,etc no entry is needed. If they contain a Separator like ML001-A1, enter only the separator -")
        other_ref = st.text_input("Other references", value=st.session_state.other_ref, help="If you have other references you want to assign, type in well location, separate by using commas (,).")
        scientific_notation = st.checkbox("Scientific notation", value=True, help="If checked, the values will be displayed in scientific notation.")

    
    # Submit button and processing
    if st.button("Submit for Analysis"):
        st.session_state.active_tab = "Classic analysis"
        if uploaded_file is not None and product_column:
            try:
                df, data = process_excel_file(uploaded_file, product_column, separator, substrate_column, references,use_conversion)


                # Create the heatmap
                fig, results = create_heatmap(data, references, neg_control, other_ref, digits, scientific_notation)

                #change session_states
                st.session_state.product_column = product_column
                st.session_state.references = references
                st.session_state.neg_control = neg_control
                st.session_state.substrate_column = substrate_column
                st.session_state.separator = separator
                st.session_state.other_ref = other_ref
                st.session_state.use_conversion = use_conversion
                st.session_state.digits = digits
                st.session_state.scientific_notation = scientific_notation
                


                # Display the heatmap using columns for better layout
                col1p, col2p, col3p = st.columns([1,3,1])
                with col2p:
                    st.pyplot(fig, use_container_width=True)

                name = uploaded_file.name.split(".")[0]
                fig_name = f"{name}_heatmap.png"
                data_name = f"{name}_results.xlsx"

                # Save the figure to a BytesIO object instead of a file
                fig_buffer = io.BytesIO()
                fig.savefig(fig_buffer, format='png', bbox_inches="tight")
                fig_buffer.seek(0)

                # Save the data to a BytesIO object instead of a file
                data_buffer = io.BytesIO()
                results.to_excel(data_buffer, index=False)
                data_buffer.seek(0)

                # Create a dictionary with file names and their corresponding data
                files_to_zip = {
                    fig_name: fig_buffer.getvalue(),
                    data_name: data_buffer.getvalue()
                }

                # Create the zip file
                zip_data = create_zip_file(files_to_zip)

                # Create a single download button for both files
                st.download_button(
                    label="Download heatmap and results",
                    data=zip_data,
                    file_name=f"{name}_heatmap_and_results.zip",
                    mime="application/zip"
                )


                st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please upload an Excel file and specify the Product Column.")
    
with tab2:

    if uploaded_file is None:
        st.warning("Please upload an Excel file in the Classic Analysis tab first.")

    st.subheader("Comparison of two Areas including plot")
    st.info("Select the two columns you want to compare. The difference and ratio will be calculated for each row. The plot shows the Ratio of the two columns in a 96-well plate format.")
    col1, col2 = st.columns(2)
    
    with col1:
        compare_1 = st.text_input("Compare_1", value=st.session_state.compare_1, help="First column for comparison")
        compare_2 = st.text_input("Compare_2", value=st.session_state.compare_2, help="Second column for comparison")

    if st.button("Submit for Comparison"):
        st.session_state.active_tab = "Advanced analysis"
        if uploaded_file is not None and compare_1 != '' and compare_2 != '':
            
            st.session_state.comparison_result = compare_columns(uploaded_file, compare_1, compare_2)
            st.write("Comparison Result:")
            st.dataframe(st.session_state.comparison_result)
    
            # Reshape the data to fit a 96-well plate format
            heatmap_data = st.session_state.comparison_result['Ratio']

            # Create the heatmap using the create_heatmap function
            fig, _ = create_heatmap(heatmap_data, references, neg_control, other_ref, digits, scientific_notation)

            # Display the heatmap
            col1p, col2p, col3p = st.columns([1,3,1])
            with col2p:
                st.pyplot(fig, use_container_width=True)
      

            #zip the heatmap_data and plot and download
            name = uploaded_file.name.split(".")[0]
            fig_name = f"{name}_comparison_heatmap.png"
            data_name = f"{name}_comparison_results.xlsx"

            # Save the figure to a BytesIO object instead of a file
            fig_buffer = io.BytesIO()
            fig.savefig(fig_buffer, format='png', bbox_inches="tight")
            fig_buffer.seek(0)

            # Save the data to a BytesIO object instead of a file
            data_buffer = io.BytesIO()
            st.session_state.comparison_result.to_excel(data_buffer, index=False)
            data_buffer.seek(0)

            # Create a dictionary with file names and their corresponding data
            files_to_zip = {
                fig_name: fig_buffer.getvalue(),
                data_name: data_buffer.getvalue()
            }

            # Create the zip file
            zip_data = create_zip_file(files_to_zip)

            # Create a single download button for both files
            st.download_button(
                label="Download comparison heatmap and results",
                data=zip_data,
                file_name=f"{name}_comparison_heatmap_and_results.zip",
                mime="application/zip"
                )

            st.success("Comparison completed successfully!")
            st.session_state.compare_1 = compare_1
            st.session_state.compare_2 = compare_2

    

    st.subheader("Selectivity of two products")
    st.info("Selectivity is calculated as (A1-A2)/(A1+A2) where A1 and A2 are the areas of the two products. Select two columns to calculate selectivtiy. Plotly is used for interactive plotting.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        select_area1 = st.text_input("Select_Area1", value=st.session_state.select_area1, help="First column for selectivity calculation")
        select_area2 = st.text_input("Select_Area2", value=st.session_state.select_area2, help="Second column for selectivity calculation")

    with col2:
        use_plotly = st.checkbox("Plotly plot", value=st.session_state.use_plotly, help="Use Plotly for interactive plotting")


    if st.button("Calculate Selectivity"):
        st.session_state.active_tab = "Advanced analysis"
        if uploaded_file is not None and select_area1 != '' and select_area2 != '':
            st.session_state.selectivity_result = calculate_selectivity(uploaded_file, select_area1, select_area2)
            st.write("Selectivity Result:")
            st.dataframe(st.session_state.selectivity_result)
            
            if use_plotly:
                fig = go.Figure()
                #plot Area1 + Area2 on x axis and selectivity (Area1 vs Area2) on y axis
                hover_text = st.session_state.selectivity_result['A']
                fig.add_trace(go.Scatter(x=st.session_state.selectivity_result[select_area1] + st.session_state.selectivity_result[select_area2], y=st.session_state.selectivity_result['Selectivity'], mode='markers', name='Selectivity', text=hover_text, hoverinfo='text'))
                fig.update_layout(xaxis_title='Area1 + Area2', yaxis_title='Selectivity (Area1 vs Area2)')
                #change width and height of the plot
                fig.update_layout(width=1000, height=600)
                col1p, col2p, col3p = st.columns([1,3,1])
                with col2p:
                    st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.selectivity_result[select_area1] + st.session_state.selectivity_result[select_area2], st.session_state.selectivity_result['Selectivity'])
                ax.set_xlabel("Area1 + Area2")
                ax.set_ylabel("Selectivity (Area1 vs Area2)")
                col1p, col2p, col3p = st.columns([1,3,1])
                with col2p:
                    st.pyplot(fig, use_container_width=True)


            #zip the heatmap_data and plot and download
            name = uploaded_file.name.split(".")[0]
            
            data_name = f"{name}_selectivity_results.xlsx"
            # Save the figure to a BytesIO object instead of a file
            fig_buffer = io.BytesIO()
            #plotly plot can't be saved as png but as html
            if use_plotly:
                fig_name = f"{name}_selectivity_plot.html"
                html_str = fig.to_html()
                fig_buffer = io.BytesIO(html_str.encode())
            else:
                fig_name = f"{name}_selectivity_plot.png"
                fig.savefig(fig_buffer, format='png', bbox_inches="tight")
            fig_buffer.seek(0)

            # Save the data to a BytesIO object instead of a file
            data_buffer = io.BytesIO()
            st.session_state.selectivity_result.to_excel(data_buffer, index=False)
            data_buffer.seek(0)

            # Create a dictionary with file names and their corresponding data
            files_to_zip = {
                fig_name: fig_buffer.getvalue(),
                data_name: data_buffer.getvalue()
            }

            # Create the zip file
            zip_data = create_zip_file(files_to_zip)

            # Create a single download button for both files
            st.download_button(
                label="Download selectivity plot and results",
                data=zip_data,
                file_name=f"{name}_selectivity_plot_and_results.zip",
                mime="application/zip"
                )

            st.success("Selectivity calculation completed successfully!")
            st.session_state.select_area1 = select_area1
            st.session_state.select_area2 = select_area2
            st.session_state.use_plotly = use_plotly
                
        else:
            st.warning("Please upload an Excel file in the Classic Analysis tab and specify columns for selectivity calculation.")



# Custom CSS to improve the layout
st.markdown("""
<style>

/* Style for the product column input */
    div[data-testid="stTextInput"] label:has(+ div input[aria-label="Product Column [MANDATORY]"]) {
        color: red !important;
    }
    .stButton button {
        width: 100%;
    }
    .stSelectbox {
        width: 100%;
    }
    .css-1d391kg {
        width: 100%;
    }
    
</style>
""", unsafe_allow_html=True)