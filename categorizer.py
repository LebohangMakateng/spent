"""
Smart Categorizer for Bank Transactions - Spent Money Detective
Rules-based categorization system to answer "Where did my money go?"
"""

import pandas as pd
import re
from typing import Dict, List, Tuple

class TransactionCategorizer:
    """Categorize bank transactions using keyword-based rules."""
    
    def __init__(self):
        self.categories = self._define_categories()
        self.category_stats = {}
    
    def _define_categories(self) -> Dict[str, List[str]]:
        """Define comprehensive category rules based on South African merchants and patterns."""
        return {
            'Groceries & Food': [
                'checkers', 'pick n pay', 'woolworths', 'spar', 'shoprite', 'food lover',
                'dischem', 'clicks', 'pnp', 'game stores', 'makro', 'fruit & veg',
                'liquorshop', 'bottle store', 'wine', 'liquor'
            ],
            
            'Restaurants & Takeaways': [
                'uber eats', 'mr d food', 'nandos', 'kfc', 'mcdonald', 'steers', 'burger king',
                'pizza', 'romans pizza', 'debonairs', 'starbucks', 'krispy kreme', 'mugg & bean',
                'vida e caffe', 'seattle coffee', 'restaurant', 'cafe', 'bistro', 'eatery',
                'eat fresh', 'food court', 'takeaway'
            ],
            
            'Transport & Fuel': [
                'uber', 'bolt', 'taxify', 'shell', 'bp', 'caltex', 'engen', 'sasol', 'total',
                'petrol', 'fuel', 'garage', 'service station', 'parking', 'toll', 'e-toll',
                'gautrain', 'metrobus', 'taxi', 'transport'
            ],
            
            'Entertainment & Subscriptions': [
                'netflix', 'dstv', 'showmax', 'spotify', 'apple music', 'youtube', 'amazon prime',
                'disney', 'hbo', 'multichoice', 'cinema', 'movies', 'theatre', 'concert',
                'entertainment', 'subscription', 'streaming', 'music', 'games', 'gaming'
            ],
            
            'Shopping & Retail': [
                'takealot', 'amazon', 'ebay', 'zando', 'superbalist', 'spree', 'loot',
                'edgars', 'truworths', 'woolworths', 'h&m', 'zara', 'cotton on', 'mr price',
                'ackermans', 'pep', 'jet', 'shopify', 'online', 'retail', 'clothing',
                'fashion', 'shoes', 'accessories'
            ],
            
            'Utilities & Bills': [
                'electricity', 'eskom', 'city power', 'water', 'municipal', 'rates',
                'internet', 'fiber', 'adsl', 'telkom', 'mtn', 'vodacom', 'cell c',
                'rain', 'afrihost', 'webafrica', 'axxess', 'prepaid', 'airtime',
                'data', 'phone', 'mobile', 'utility', 'council', 'municipality',
                'rent', 'rental', 'lease', 'accommodation', 'housing', 'property'
            ],
            
            'Banking & Fees': [
                'bank charge', 'service fee', 'monthly fee', 'transaction fee', 'atm fee',
                'overdraft', 'interest', 'admin fee', 'account fee', 'card fee',
                'fnb', 'absa', 'standard bank', 'nedbank', 'capitec', 'african bank',
                'byc debit', 'magtape'
            ],
            
            'Health & Pharmacy': [
                'clicks', 'dischem', 'pharmacy', 'chemist', 'medical', 'doctor', 'dentist',
                'hospital', 'clinic', 'health', 'medicine', 'prescription', 'wellness'
            ],
            
            'Transfers & Payments': [
                'transfer', 'payment', 'pmt', 'rtc', 'eft', 'payshap', 'instant pay',
                'beneficiary', 'recipient', 'person to person', 'p2p'
            ],
            
            'Income & Credits': [
                'salary', 'wage', 'income', 'deposit', 'credit', 'refund', 'cashback',
                'interest', 'dividend', 'bonus', 'commission', 'freelance', 'consulting'
            ],
            
            'Investments & Savings': [
                'investment', 'invest', 'save', 'savings', 'unit trust', 'mutual fund',
                'shares', 'stocks', 'bonds', 'retirement', 'pension', 'provident',
                'tfsa', 'tax free', 'fixed deposit', 'money market'
            ]
        }
    
    def categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize a single transaction based on description and amount."""
        if pd.isna(description) or description == '':
            return 'Uncategorized'
        
        description_lower = description.lower().strip()
        
        # Handle income transactions (positive amounts)
        if amount > 0:
            # Check if it's a specific income type
            for keyword in self.categories['Income & Credits']:
                if keyword in description_lower:
                    return 'Income & Credits'
            
            # Check if it's a transfer or refund
            for keyword in self.categories['Transfers & Payments']:
                if keyword in description_lower:
                    return 'Transfers & Payments'
            
            # Default positive amounts to Income
            return 'Income & Credits'
        
        # Handle expense transactions (negative amounts)
        # First, check for high-priority categories (more specific keywords)
        high_priority_categories = ['Utilities & Bills', 'Groceries & Food', 'Restaurants & Takeaways', 'Transport & Fuel', 'Investments & Savings']
        
        for category in high_priority_categories:
            for keyword in self.categories[category]:
                if keyword in description_lower:
                    return category
        
        # Then check remaining categories
        for category, keywords in self.categories.items():
            if category in ['Income & Credits'] or category in high_priority_categories:  # Skip already checked categories
                continue
                
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        # Special handling for common patterns
        if any(word in description_lower for word in ['pos purchase', 'purchase']):
            # Try to extract merchant name after "pos purchase"
            merchant_match = re.search(r'pos purchase\s+(.+?)\s+\d', description_lower)
            if merchant_match:
                merchant = merchant_match.group(1).strip()
                # Re-categorize based on merchant name
                for category, keywords in self.categories.items():
                    if category in ['Income & Credits']:
                        continue
                    for keyword in keywords:
                        if keyword in merchant:
                            return category
        
        return 'Uncategorized'
    
    def categorize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize all transactions in a DataFrame."""
        df_copy = df.copy()
        
        # Apply categorization
        df_copy['Category'] = df_copy.apply(
            lambda row: self.categorize_transaction(row['Description'], row['Amount']), 
            axis=1
        )
        
        # Generate category statistics
        self._generate_category_stats(df_copy)
        
        return df_copy
    
    def _generate_category_stats(self, df: pd.DataFrame):
        """Generate statistics about categorization performance."""
        total_transactions = len(df)
        categorized = len(df[df['Category'] != 'Uncategorized'])
        
        self.category_stats = {
            'total_transactions': total_transactions,
            'categorized_transactions': categorized,
            'uncategorized_transactions': total_transactions - categorized,
            'categorization_rate': (categorized / total_transactions) * 100 if total_transactions > 0 else 0,
            'category_breakdown': df['Category'].value_counts().to_dict()
        }
    
    def get_category_summary(self, df: pd.DataFrame) -> Dict:
        """Get spending summary by category."""
        if 'Category' not in df.columns:
            raise ValueError("DataFrame must have 'Category' column. Run categorize_dataframe first.")
        
        # Separate expenses and income
        expenses = df[df['Amount'] < 0].copy()
        income = df[df['Amount'] > 0].copy()
        
        # Calculate spending by category (make amounts positive for readability)
        expenses['Amount_Positive'] = expenses['Amount'].abs()
        category_spending = expenses.groupby('Category')['Amount_Positive'].agg(['sum', 'count', 'mean']).round(2)
        category_spending.columns = ['Total_Spent', 'Transaction_Count', 'Average_Amount']
        category_spending = category_spending.sort_values('Total_Spent', ascending=False)
        
        # Calculate income by category
        income_by_category = income.groupby('Category')['Amount'].agg(['sum', 'count']).round(2)
        income_by_category.columns = ['Total_Income', 'Transaction_Count']
        
        # Top merchants/descriptions by spending
        top_merchants = expenses.groupby('Description')['Amount_Positive'].sum().sort_values(ascending=False).head(10)
        
        summary = {
            'total_spent': expenses['Amount'].sum(),  # This will be negative
            'total_income': income['Amount'].sum(),
            'net_change': df['Amount'].sum(),
            'spending_by_category': category_spending.to_dict('index'),
            'income_by_category': income_by_category.to_dict('index'),
            'top_merchants': top_merchants.to_dict(),
            'categorization_stats': self.category_stats
        }
        
        return summary
    
    def get_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate human-readable insights about spending patterns."""
        if 'Category' not in df.columns:
            raise ValueError("DataFrame must have 'Category' column. Run categorize_dataframe first.")
        
        insights = []
        summary = self.get_category_summary(df)
        
        # Top spending category
        spending_by_cat = summary['spending_by_category']
        if spending_by_cat:
            top_category = max(spending_by_cat.keys(), key=lambda k: spending_by_cat[k]['Total_Spent'])
            top_amount = spending_by_cat[top_category]['Total_Spent']
            insights.append(f"Your biggest expense category was {top_category} (R{top_amount:,.2f})")
        
        # Top merchant
        top_merchants = summary['top_merchants']
        if top_merchants:
            top_merchant = list(top_merchants.keys())[0]
            top_merchant_amount = list(top_merchants.values())[0]
            insights.append(f"You spent the most at {top_merchant} (R{top_merchant_amount:,.2f})")
        
        # Categorization rate
        cat_rate = summary['categorization_stats']['categorization_rate']
        insights.append(f"Successfully categorized {cat_rate:.1f}% of your transactions")
        
        # Net spending
        net_change = summary['net_change']
        if net_change < 0:
            insights.append(f"You spent R{abs(net_change):,.2f} more than you earned this period")
        else:
            insights.append(f"You saved R{net_change:,.2f} this period")
        
        return insights

def categorize_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """Convenience function to categorize transactions and get insights."""
    categorizer = TransactionCategorizer()
    categorized_df = categorizer.categorize_dataframe(df)
    summary = categorizer.get_category_summary(categorized_df)
    insights = categorizer.get_insights(categorized_df)
    
    return categorized_df, summary, insights

if __name__ == "__main__":
    # Test with parsed data
    from parser import parse_bank_statement
    
    try:
        df, _ = parse_bank_statement("sample_data/sample_bank_statement.csv")
        categorized_df, summary, insights = categorize_transactions(df)
        
        print("Successfully categorized transactions!")
        print(f"Categorization rate: {summary['categorization_stats']['categorization_rate']:.1f}%")
        print("\nTop insights:")
        for insight in insights:
            print(f"- {insight}")
        
        print(f"\nFirst 5 categorized transactions:")
        print(categorized_df[['Date', 'Description', 'Amount', 'Category']].head())
        
    except Exception as e:
        print(f"Error testing categorizer: {e}")