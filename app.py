"""
Spent - Your Money Detective
Streamlit app that answers "Where did my money go?"
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import traceback
import os

# Import our modules
from parser import parse_bank_statement, BankStatementParser
from categorizer import categorize_transactions
from insights import create_all_visualizations

# Page configuration
st.set_page_config(
    page_title="Spent - Your Money Detective 💰",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and mobile responsiveness
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .insight-box {
        background: #8291a1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
        color: #000000 !important;
        font-weight: 500;
    }
    
    .insight-box p {
        color: #000000 !important;
        margin: 0;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1rem 0;
        }
        
        .subtitle {
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Force single column layout on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }
        
        /* Stack metrics vertically on mobile */
        [data-testid="metric-container"] {
            margin-bottom: 1rem;
        }
    }
    
    /* Ensure charts are properly sized on all devices */
    .js-plotly-plot {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Spent - Your Money Detective 💰</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload your bank statement and discover where your money went!</p>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your Bank Statement")
    uploaded_file = st.file_uploader(
        "Choose a CSV file from your bank",
        type=['csv'],
        help="Upload your bank statement in CSV format. Most banks allow you to export statements as CSV files."
    )
    
    if uploaded_file is not None:
        # Enforce max upload size (5 MB)
        if getattr(uploaded_file, "size", 0) and uploaded_file.size > 5 * 1024 * 1024:
            st.error("File too large. Please upload a CSV under 5 MB.")
            return

        try:
            # Show processing message
            with st.spinner('🔍 Analyzing your spending patterns...'):
                # Parse the uploaded file
                df = parse_uploaded_file(uploaded_file)
                
                # Categorize transactions
                categorized_df, summary, category_insights = categorize_transactions(df)
                
                # Generate visualizations
                visualizations = create_all_visualizations(categorized_df)
            
            # Success message
            st.success(f"✅ Successfully analyzed {len(categorized_df)} transactions!")
            
            # Display results
            display_money_detective_dashboard(categorized_df, visualizations)
            
            # Download section
            display_download_section(categorized_df)
            
        except Exception as e:
            st.error(f"❌ Error processing your bank statement: {str(e)}")
            st.error("Please make sure your CSV file has columns for Date, Description, Amount, and Balance.")
            
            # Show debug info in expander (only in debug mode)
            if os.getenv("SPENT_DEBUG") == "1":
                with st.expander("Debug Information"):
                    st.code(traceback.format_exc())
    
    else:
        # Show sample data and instructions
        display_instructions()

def parse_uploaded_file(uploaded_file):
    """Parse the uploaded CSV file."""
    # Read the file content
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Create a temporary file-like object for our parser
    parser = BankStatementParser()
    
    # Load CSV from string
    try:
        df = pd.read_csv(stringio, encoding='utf-8')
    except UnicodeDecodeError:
        stringio.seek(0)
        df = pd.read_csv(stringio, encoding='utf-8-sig')
    
    # Clean and normalize
    cleaned_df = parser.clean_and_normalize(df)
    
    return cleaned_df

def display_money_detective_dashboard(df, visualizations):
    """Display the main money detective dashboard."""
    
    # Summary cards
    st.markdown("### 💡 Your Money at a Glance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cards = visualizations['summary_cards']
    
    with col1:
        st.metric(
            label=cards['total_spent']['title'],
            value=cards['total_spent']['value'],
            help=cards['total_spent']['subtitle']
        )
    
    with col2:
        st.metric(
            label=cards['biggest_category']['title'],
            value=cards['biggest_category']['value'],
            help=cards['biggest_category']['subtitle']
        )
    
    with col3:
        st.metric(
            label=cards['daily_average']['title'],
            value=cards['daily_average']['value'],
            help=cards['daily_average']['subtitle']
        )
    
    with col4:
        st.metric(
            label=cards['top_merchant']['title'],
            value=cards['top_merchant']['value'],
            help=cards['top_merchant']['subtitle']
        )
    
    # Key insights
    st.markdown("### 🎯 Key Insights")
    for insight in visualizations['text_insights']:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Charts section
    st.markdown("### 📊 Visual Breakdown")
    
    # Display charts vertically one after another
    st.markdown("#### 💰 Category Breakdown")
    st.plotly_chart(visualizations['pie_chart'], use_container_width=True)
    
    st.markdown("#### 🏪 Top Spending")
    st.plotly_chart(visualizations['bar_chart'], use_container_width=True)
    
    st.markdown("#### 📈 Daily Trends")
    st.plotly_chart(visualizations['line_chart'], use_container_width=True)
    
    st.markdown("#### 📅 Weekly Pattern")
    st.plotly_chart(visualizations['weekly_chart'], use_container_width=True)
    
    # Transaction table
    with st.expander("📋 View All Categorized Transactions"):
        st.dataframe(
            df[['Date', 'Description', 'Amount', 'Category']].sort_values('Date', ascending=False),
            use_container_width=True
        )

def display_download_section(df):
    """Display download options for processed data."""
    st.markdown("### 💾 Download Your Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download categorized CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Categorized Transactions (CSV)",
            data=csv_data,
            file_name="spent_categorized_transactions.csv",
            mime="text/csv",
            help="Download your transactions with categories added"
        )
    
    with col2:
        # Download summary report
        summary_report = generate_summary_report(df)
        st.download_button(
            label="📊 Download Summary Report (TXT)",
            data=summary_report,
            file_name="spent_summary_report.txt",
            mime="text/plain",
            help="Download a text summary of your spending analysis"
        )

def generate_summary_report(df):
    """Generate a text summary report."""
    from insights import MoneyDetectiveInsights
    
    detector = MoneyDetectiveInsights()
    insights_data = detector.generate_spending_insights(df)
    text_insights = detector.generate_text_insights(df)
    
    report = f"""
SPENT - YOUR MONEY DETECTIVE REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY ===
Total Spent: R{insights_data['summary']['total_spent']:,.2f}
Total Income: R{insights_data['summary']['total_income']:,.2f}
Net Change: R{insights_data['summary']['net_change']:,.2f}
Daily Average Spending: R{insights_data['summary']['daily_average_spending']:,.2f}

=== KEY INSIGHTS ===
"""
    
    for insight in text_insights:
        report += f"• {insight}\n"
    
    report += "\n=== SPENDING BY CATEGORY ===\n"
    for category, data in insights_data['category_breakdown'].items():
        report += f"{category}: R{data['Total_Spent']:,.2f} ({data['Transaction_Count']} transactions)\n"
    
    report += "\n=== TOP MERCHANTS ===\n"
    for merchant, amount in list(insights_data['top_merchants'].items())[:10]:
        report += f"{merchant}: R{amount:,.2f}\n"
    
    return report

def display_instructions():
    """Display instructions and sample data info."""
    st.markdown("### 🚀 How to Get Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Export Your Bank Statement**
        1. Log into your online banking
        2. Go to your account statements
        3. Export/Download as CSV format
        4. Save the file to your computer
        """)
    
    with col2:
        st.markdown("""
        **Step 2: Upload & Analyze**
        1. Click "Choose a CSV file" above
        2. Select your bank statement CSV
        3. Wait for the magic to happen! ✨
        4. Discover where your money went
        """)
    
    st.markdown("### 🏦 Supported Banks")
    st.info("Spent works with CSV exports from most South African banks including FNB, ABSA, Standard Bank, Nedbank, and Capitec. The app automatically detects and normalizes different CSV formats.")
    
    # Sample data demo
    if st.button("🎮 Try with Sample Data"):
        try:
            with st.spinner('Loading sample data...'):
                df, _ = parse_bank_statement("sample_data/sample_bank_statement.csv")
                categorized_df, _, _ = categorize_transactions(df)
                visualizations = create_all_visualizations(categorized_df)
            
            st.success("✅ Sample data loaded!")
            display_money_detective_dashboard(categorized_df, visualizations)
            
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

if __name__ == "__main__":
    main()