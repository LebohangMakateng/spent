"""
Insights Generator & Visualization Suite - Spent Money Detective
Creates visual insights to answer "Where did my money go?"
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class MoneyDetectiveInsights:
    """Generate visual insights and charts for spending analysis."""
    
    def __init__(self):
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def generate_spending_insights(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive spending insights."""
        if 'Category' not in df.columns:
            raise ValueError("DataFrame must have 'Category' column")
        
        # Separate expenses and income
        expenses = df[df['Amount'] < 0].copy()
        income = df[df['Amount'] > 0].copy()
        
        # Basic metrics
        total_spent = abs(expenses['Amount'].sum())
        total_income = income['Amount'].sum()
        net_change = df['Amount'].sum()
        
        # Category analysis
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        category_spending = expenses.groupby('Category')['Amount_Positive'].agg(['sum', 'count', 'mean']).round(2)
        category_spending.columns = ['Total_Spent', 'Transaction_Count', 'Average_Amount']
        category_spending = category_spending.sort_values('Total_Spent', ascending=False)
        
        # Top merchants
        top_merchants = expenses.groupby('Description')['Amount_Positive'].sum().sort_values(ascending=False).head(10)
        
        # Daily spending analysis
        expenses['Date'] = pd.to_datetime(expenses['Date'])
        daily_spending = expenses.groupby(expenses['Date'].dt.date)['Amount_Positive'].sum()
        
        # Weekly patterns
        expenses['DayOfWeek'] = expenses['Date'].dt.day_name()
        weekly_pattern = expenses.groupby('DayOfWeek')['Amount_Positive'].mean()
        
        # Spending velocity
        date_range = (expenses['Date'].max() - expenses['Date'].min()).days
        daily_average = total_spent / max(date_range, 1)
        
        insights = {
            'summary': {
                'total_spent': total_spent,
                'total_income': total_income,
                'net_change': net_change,
                'daily_average_spending': daily_average,
                'total_transactions': len(df),
                'expense_transactions': len(expenses),
                'income_transactions': len(income)
            },
            'category_breakdown': category_spending.to_dict('index'),
            'top_merchants': top_merchants.to_dict(),
            'daily_spending': daily_spending.to_dict(),
            'weekly_pattern': weekly_pattern.to_dict(),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d') if pd.notna(df['Date'].min()) else 'Unknown',
                'end': df['Date'].max().strftime('%Y-%m-%d') if pd.notna(df['Date'].max()) else 'Unknown'
            }
        }
        
        return insights
    
    def create_category_pie_chart(self, df: pd.DataFrame, title: str = "Where Did Your Money Go?") -> go.Figure:
        """Create pie chart showing spending breakdown by category."""
        expenses = df[df['Amount'] < 0].copy()
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        
        category_totals = expenses.groupby('Category')['Amount_Positive'].sum().sort_values(ascending=False)
        
        # Filter out very small categories (less than 1% of total)
        total_spending = category_totals.sum()
        significant_categories = category_totals[category_totals >= total_spending * 0.01]
        
        fig = px.pie(
            values=significant_categories.values,
            names=significant_categories.index,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: R%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            font=dict(size=12),
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_top_merchants_bar_chart(self, df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create bar chart showing top spending destinations."""
        expenses = df[df['Amount'] < 0].copy()
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        
        top_merchants = expenses.groupby('Description')['Amount_Positive'].sum().sort_values(ascending=False).head(top_n)
        
        # Clean up merchant names for display
        merchant_names = [name[:50] + '...' if len(name) > 50 else name for name in top_merchants.index]
        
        # Create a DataFrame for Plotly
        chart_data = pd.DataFrame({
            'Merchant': merchant_names,
            'Amount': top_merchants.values
        })
        
        fig = px.bar(
            chart_data,
            x='Amount',
            y='Merchant',
            orientation='h',
            title=f"Top {top_n} Spending Destinations",
            labels={'Amount': 'Amount Spent (R)', 'Merchant': 'Merchant'},
            color='Amount',
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Amount: R%{x:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            height=max(400, top_n * 40),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            coloraxis_showscale=False
        )
        
        return fig
    
    def create_spending_trend_line_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create line chart showing daily spending trends."""
        expenses = df[df['Amount'] < 0].copy()
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        expenses['Date'] = pd.to_datetime(expenses['Date'])
        
        # Daily spending
        daily_spending = expenses.groupby(expenses['Date'].dt.date)['Amount_Positive'].sum().reset_index()
        daily_spending.columns = ['Date', 'Amount']
        
        # Calculate 7-day moving average
        daily_spending = daily_spending.sort_values('Date')
        daily_spending['Moving_Average'] = daily_spending['Amount'].rolling(window=7, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Add daily spending line
        fig.add_trace(go.Scatter(
            x=daily_spending['Date'],
            y=daily_spending['Amount'],
            mode='lines+markers',
            name='Daily Spending',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='<b>%{x}</b><br>Spent: R%{y:,.2f}<extra></extra>'
        ))
        
        # Add moving average line
        fig.add_trace(go.Scatter(
            x=daily_spending['Date'],
            y=daily_spending['Moving_Average'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#4ECDC4', width=3, dash='dash'),
            hovertemplate='<b>%{x}</b><br>7-Day Avg: R%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Daily Spending Trends",
            xaxis_title="Date",
            yaxis_title="Amount Spent (R)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_weekly_pattern_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create chart showing spending patterns by day of week."""
        expenses = df[df['Amount'] < 0].copy()
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        expenses['Date'] = pd.to_datetime(expenses['Date'])
        expenses['DayOfWeek'] = expenses['Date'].dt.day_name()
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_spending = expenses.groupby('DayOfWeek')['Amount_Positive'].mean().reindex(day_order, fill_value=0)
        
        # Create a DataFrame for Plotly
        chart_data = pd.DataFrame({
            'Day': day_order,
            'Amount': weekly_spending.values
        })
        
        fig = px.bar(
            chart_data,
            x='Day',
            y='Amount',
            title="Average Spending by Day of Week",
            labels={'Day': 'Day of Week', 'Amount': 'Average Amount (R)'},
            color='Amount',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Average: R%{y:,.2f}<extra></extra>'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        
        return fig
    
    def create_summary_cards_data(self, df: pd.DataFrame) -> Dict:
        """Create data for summary cards display."""
        insights = self.generate_spending_insights(df)
        summary = insights['summary']
        
        # Find biggest category
        category_breakdown = insights['category_breakdown']
        biggest_category = max(category_breakdown.keys(), key=lambda k: category_breakdown[k]['Total_Spent']) if category_breakdown else "Unknown"
        biggest_category_amount = category_breakdown[biggest_category]['Total_Spent'] if category_breakdown else 0
        
        # Find most frequent merchant
        top_merchants = insights['top_merchants']
        most_frequent_merchant = list(top_merchants.keys())[0] if top_merchants else "Unknown"
        most_frequent_amount = list(top_merchants.values())[0] if top_merchants else 0
        
        cards_data = {
            'total_spent': {
                'title': 'Total Spent',
                'value': f"R{summary['total_spent']:,.2f}",
                'subtitle': f"across {summary['expense_transactions']} transactions"
            },
            'biggest_category': {
                'title': 'Biggest Expense',
                'value': biggest_category,
                'subtitle': f"R{biggest_category_amount:,.2f}"
            },
            'daily_average': {
                'title': 'Daily Average',
                'value': f"R{summary['daily_average_spending']:,.2f}",
                'subtitle': "spending per day"
            },
            'top_merchant': {
                'title': 'Top Destination',
                'value': most_frequent_merchant[:30] + '...' if len(most_frequent_merchant) > 30 else most_frequent_merchant,
                'subtitle': f"R{most_frequent_amount:,.2f}"
            }
        }
        
        return cards_data
    
    def generate_text_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate human-readable insights about spending patterns."""
        insights_data = self.generate_spending_insights(df)
        summary = insights_data['summary']
        category_breakdown = insights_data['category_breakdown']
        
        insights = []
        
        # Total spending insight
        insights.append(f"You spent R{summary['total_spent']:,.2f} over {summary['expense_transactions']} transactions")
        
        # Top category insight
        if category_breakdown:
            top_category = max(category_breakdown.keys(), key=lambda k: category_breakdown[k]['Total_Spent'])
            top_amount = category_breakdown[top_category]['Total_Spent']
            percentage = (top_amount / summary['total_spent']) * 100
            insights.append(f"Your biggest expense was {top_category} (R{top_amount:,.2f} - {percentage:.1f}% of total)")
        
        # Daily average insight
        insights.append(f"You spent an average of R{summary['daily_average_spending']:,.2f} per day")
        
        # Net change insight
        if summary['net_change'] < 0:
            insights.append(f"You spent R{abs(summary['net_change']):,.2f} more than you earned")
        else:
            insights.append(f"You saved R{summary['net_change']:,.2f} this period")
        
        # Weekly pattern insight
        weekly_pattern = insights_data['weekly_pattern']
        if weekly_pattern:
            highest_day = max(weekly_pattern.keys(), key=lambda k: weekly_pattern[k])
            insights.append(f"You tend to spend most on {highest_day}s (R{weekly_pattern[highest_day]:,.2f} average)")
        
        return insights

def create_all_visualizations(df: pd.DataFrame) -> Dict:
    """Create all visualizations and insights for the money detective dashboard."""
    detector = MoneyDetectiveInsights()
    
    visualizations = {
        'pie_chart': detector.create_category_pie_chart(df),
        'bar_chart': detector.create_top_merchants_bar_chart(df),
        'line_chart': detector.create_spending_trend_line_chart(df),
        'weekly_chart': detector.create_weekly_pattern_chart(df),
        'summary_cards': detector.create_summary_cards_data(df),
        'text_insights': detector.generate_text_insights(df),
        'raw_insights': detector.generate_spending_insights(df)
    }
    
    return visualizations

if __name__ == "__main__":
    # Test with categorized data
    from parser import parse_bank_statement
    from categorizer import categorize_transactions
    
    try:
        df, _ = parse_bank_statement("sample_data/sample_bank_statement.csv")
        categorized_df, _, _ = categorize_transactions(df)
        
        visualizations = create_all_visualizations(categorized_df)
        
        print("Successfully generated insights!")
        print("\nKey Insights:")
        for insight in visualizations['text_insights']:
            print(f"  {insight}")
        
        print(f"\nSummary Cards:")
        for card_name, card_data in visualizations['summary_cards'].items():
            print(f"  {card_data['title']}: {card_data['value']} ({card_data['subtitle']})")
        
    except Exception as e:
        print(f"Error generating insights: {e}")