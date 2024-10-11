import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load dataset
#df = pd.read_csv('/content/mutual_funds_data.csv')
# Load dataset from GitHub
df = pd.read_csv('https://github.com/vaidyamohit/Dashboard/blob/main/mutual_funds_data.csv')
df[['beta', 'sd']] = df[['beta', 'sd']].apply(pd.to_numeric, errors='coerce').dropna(subset=['beta', 'sd'])

# Create fund size group column
df['fund_size_group'] = pd.cut(df['fund_size_cr'],
                               bins=[-float('inf'), 500, 750, 2000, 5000, 10000, 50000, float('inf')],
                               labels=['0-500', '500-750', '750-2000', '2000-5000', '5000-10000', '10000-50000', '>50000'])

# Model: returns_5yr = b0 + b1*feature1 + b2*feature2 + b3*feature3 + e
lin_reg_model = smf.ols('returns_5yr ~ min_sip + min_lumpsum + expense_ratio + fund_size_cr + fund_age_yr + sortino + alpha + sd + beta + sharpe', data=df).fit()

# Section 1
st.title("Mutual Funds Analysis Dashboard")
st.header("Section 1: Mutual Funds Category Analysis")

selected_category1 = st.selectbox("Select Category", df['category'].unique())
filtered_data1 = df[df['category'] == selected_category1]

scatter_plot = px.scatter(filtered_data1, x='returns_1yr', y='returns_3yr', color='risk_level', size='fund_size_cr',
                          title=f'Scatter Plot for {selected_category1}')
st.plotly_chart(scatter_plot)

st.write("Data Table:")
st.dataframe(filtered_data1)

# Section 2
st.header("Section 2: Mutual Funds Minimum SIP and Lump Sum Analysis")

# Add unique keys for the selectboxes
selected_category2 = st.selectbox("Select Category", df['category'].unique(), key="category_selectbox2")
filtered_df2 = df[df['category'] == selected_category2]

st.subheader("Distribution of min_sip")
fig_min_sip = px.histogram(filtered_df2, x='min_sip', title='Distribution of min_sip')
st.plotly_chart(fig_min_sip)

st.subheader("Distribution of min_lumpsum")
fig_min_lumpsum = px.histogram(filtered_df2, x='min_lumpsum', title='Distribution of min_lumpsum')
st.plotly_chart(fig_min_lumpsum)

st.subheader("Distribution of min_sip and min_lumpsum by category type")
fig_min_by_category = px.box(filtered_df2, x='category', y=['min_sip', 'min_lumpsum'],
                              title='Distribution of min_sip and min_lumpsum by category type')
st.plotly_chart(fig_min_by_category)

st.subheader("Distribution of min_sip and min_lumpsum by sub_category type")
fig_min_by_sub_category = px.box(filtered_df2, x='sub_category', y=['min_sip', 'min_lumpsum'],
                                 title='Distribution of min_sip and min_lumpsum by sub_category type')
st.plotly_chart(fig_min_by_sub_category)

st.write("Data Table:")
st.dataframe(filtered_df2)


# Section 3
st.header("Section 3: Mutual Funds Expense Ratio Analysis")

# Add unique keys for the selectboxes
selected_category3 = st.selectbox("Select Category", df['category'].unique(), key="category_selectbox3")
filtered_df3 = df[df['category'] == selected_category3]

fig_fund_size_vs_expense_ratio = px.bar(filtered_df3, x='fund_size_group', y='expense_ratio', color='category',
                                        barmode='group', title='Fund Size vs Expense Ratio by Category')
st.plotly_chart(fig_fund_size_vs_expense_ratio)

fig_expense_ratio_distribution = make_subplots(rows=1, cols=2, subplot_titles=['Histogram', 'Box Plot'])
fig_expense_ratio_distribution.add_trace(go.Histogram(x=filtered_df3['expense_ratio'], nbinsx=20), row=1, col=1)
fig_expense_ratio_distribution.add_trace(go.Box(x=filtered_df3['expense_ratio']), row=1, col=2)
fig_expense_ratio_distribution.update_layout(title='Expense Ratio Distribution')
st.plotly_chart(fig_expense_ratio_distribution)

fig_expense_ratio_vs_rating = px.box(filtered_df3, x='rating', y='expense_ratio', title='Expense Ratio vs Rating')
st.plotly_chart(fig_expense_ratio_vs_rating)

fig_expense_ratio_vs_risk_level = px.box(filtered_df3, x='risk_level', y='expense_ratio', title='Expense Ratio vs Risk Level')
st.plotly_chart(fig_expense_ratio_vs_risk_level)

fig_expense_ratio_vs_category = px.box(df, y='expense_ratio', x='category', title='Expense Ratio vs Category')
st.plotly_chart(fig_expense_ratio_vs_category)

fig_expense_ratio_vs_sub_category = px.box(df, y='expense_ratio', x='sub_category', title='Expense Ratio vs Sub-Category')
st.plotly_chart(fig_expense_ratio_vs_sub_category)

# Section 4
st.header("Section 4: Mutual Funds Standard Deviation Analysis")

# Add unique keys for the selectboxes
selected_category4 = st.selectbox("Select Category", df['category'].unique(), key="category_selectbox4")
filtered_df4 = df[df['category'] == selected_category4]

fig_sd_distribution = px.histogram(filtered_df4, x='sd', nbins=30, title='Distribution of Standard Deviation')
st.plotly_chart(fig_sd_distribution)

fig_sd_vs_category = px.bar(filtered_df4, x='category', y='sd', labels={'sd': 'Standard Deviation'},
                            title='Standard Deviation Variation across Category')
st.plotly_chart(fig_sd_vs_category)

fig_sd_vs_sub_category = px.bar(filtered_df4, x='sd', y='sub_category', orientation='h',
                                labels={'sd': 'Standard Deviation'},
                                title='Standard Deviation Variation across Sub-Category')
st.plotly_chart(fig_sd_vs_sub_category)

fig_sd_vs_risk_level = px.bar(filtered_df4, x='sd', y='risk_level', orientation='h',
                              labels={'sd': 'Standard Deviation'},
                              title='Standard Deviation Variation across Risk Level')
st.plotly_chart(fig_sd_vs_risk_level)

fig_sd_vs_rating = px.bar(filtered_df4, x='rating', y='sd', labels={'sd': 'Standard Deviation'},
                          title='Standard Deviation Variation across Rating')
st.plotly_chart(fig_sd_vs_rating)

# Section 5: Linear Regression Prediction
st.header("Section 5: Linear Regression Prediction")

# Input widgets for user input with float values
min_sip = st.number_input("Min SIP", step=0.01)
min_lumpsum = st.number_input("Min Lumpsum", step=0.01)
expense_ratio = st.number_input("Expense Ratio", step=0.01)
fund_size_cr = st.number_input("Fund Size", step=0.01)
fund_age_yr = st.number_input("Fund Age", step=0.01)
sortino = st.number_input("Sortino Ratio", step=0.01)
alpha = st.number_input("Alpha", step=0.01)
sd = st.number_input("Standard Deviation", step=0.01)
beta = st.number_input("Beta", step=0.01)
sharpe = st.number_input("Sharpe Ratio", step=0.01)

# Button to trigger prediction
predict_button = st.button("Predict Returns")

# Output section for prediction result
if predict_button:
    input_data = pd.DataFrame({
        'min_sip': [min_sip],
        'min_lumpsum': [min_lumpsum],
        'expense_ratio': [expense_ratio],
        'fund_size_cr': [fund_size_cr],
        'fund_age_yr': [fund_age_yr],
        'sortino': [sortino],
        'alpha': [alpha],
        'sd': [sd],
        'beta': [beta],
        'sharpe': [sharpe]
    })

    # Predict using the linear regression model
    prediction = lin_reg_model.predict(input_data)

    st.success(f'Predicted 5-year Returns: {prediction.iloc[0]:.2f}%')

# Section 6 - Group 1: Category, Risk Level, Rating
st.header("Section 6: Scheme Analysis - Group 1")

# Dropdowns for scheme analysis - Group 1
selected_category_group1 = st.selectbox("Select Category", df['category'].unique(), key="category_selector_group1")
selected_risk_level_group1 = st.selectbox("Select Risk Level", df['risk_level'].unique(), key="risk_level_selector_group1")
selected_rating_group1 = st.selectbox("Select Rating", df['rating'].unique(), key="rating_selector_group1")

# Button to trigger scheme analysis - Group 1
display_button_scheme_group1 = st.button("Display Scheme Names - Group 1")

# Output section for scheme analysis - Group 1
if display_button_scheme_group1:
    # Filter data based on selected criteria - Group 1
    filtered_data_scheme_group1 = df[
        (df['category'] == selected_category_group1) &
        (df['risk_level'] == selected_risk_level_group1) &
        (df['rating'] == selected_rating_group1)
    ]

    if filtered_data_scheme_group1.empty:
        st.warning("No schemes available for the selected criteria in Group 1.")
    else:
        # Display the filtered data - Group 1
        scheme_table_data_group1 = {
            'Scheme Names': filtered_data_scheme_group1['scheme_name'].tolist(),
            'Returns (1 Year)': filtered_data_scheme_group1['returns_1yr'].tolist(),
        }
        scheme_table_group1 = pd.DataFrame(scheme_table_data_group1)
        st.table(scheme_table_group1)

        # Display bar graph for the first 5 schemes - Group 1
        fig_scheme_graph_group1 = px.bar(x=scheme_table_group1['Scheme Names'][:5], y=scheme_table_group1['Returns (1 Year)'][:5],
                                         labels={'x': 'Scheme Names', 'y': 'Returns (1 Year)'},
                                         title='Returns (1 Year) for Selected Schemes - Group 1')
        st.plotly_chart(fig_scheme_graph_group1)

# Define expense ratio ranges
expense_ratio_ranges = [
    (0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1),
    (1, 1.2),
    (1.2, 1.4),
    (1.4, 1.6),
    (1.6, 1.8),
    (1.8, 2),
    (2, float('inf'))  # The last range is for values greater than 2
]

# Section 6 - Group 2: Category, Min SIP, Expense Ratio
st.header("Section 6: Scheme Analysis - Group 2")

# Dropdowns for scheme analysis - Group 2
selected_category_group2 = st.selectbox("Select Category", df['category'].unique(), key="category_selector_group2")
selected_min_sip_group2 = st.selectbox("Select Minimum SIP", df['min_sip'].unique(), key="min_sip_selector_group2")
selected_expense_ratio_group2 = st.selectbox("Select Expense Ratio Range", expense_ratio_ranges, key="expense_ratio_selector_group2")

# Button to trigger scheme analysis - Group 2
display_button_scheme_group2 = st.button("Display Scheme Names - Group 2")

# Output section for scheme analysis - Group 2
if display_button_scheme_group2:
    # Filter data based on selected criteria - Group 2
    filtered_data_scheme_group2 = df[
        (df['category'] == selected_category_group2) &
        (df['min_sip'] == selected_min_sip_group2) &
        ((df['expense_ratio'] >= selected_expense_ratio_group2[0]) & (df['expense_ratio'] < selected_expense_ratio_group2[1]))
    ]

    if filtered_data_scheme_group2.empty:
        st.warning("No schemes available for the selected criteria in Group 2.")
    else:
        # Display the filtered data - Group 2
        scheme_table_data_group2 = {
            'Scheme Names': filtered_data_scheme_group2['scheme_name'].tolist(),
            'Returns (1 Year)': filtered_data_scheme_group2['returns_1yr'].tolist(),
        }
        scheme_table_group2 = pd.DataFrame(scheme_table_data_group2)
        st.table(scheme_table_group2)

        # Display bar graph for the first 5 schemes - Group 2
        fig_scheme_graph_group2 = px.bar(x=scheme_table_group2['Scheme Names'][:5], y=scheme_table_group2['Returns (1 Year)'][:5],
                                         labels={'x': 'Scheme Names', 'y': 'Returns (1 Year)'},
                                         title='Returns (1 Year) for Selected Schemes - Group 2')
        st.plotly_chart(fig_scheme_graph_group2)

