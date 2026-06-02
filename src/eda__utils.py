def plot_boxplots_outliers(df, cols):
    """Plot box plots for selected numeric columns."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col])
        axes[i].set_title(f'Box Plot – {col}')
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df, cat_cols):
    """Plot bar charts for selected categorical columns."""
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cat_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def plot_hourly_volume(df):
    """Plot transaction volume by hour."""
    hourly_counts = df.groupby('TransactionHour').size()
    plt.figure(figsize=(10, 5))
    hourly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Transaction Volume by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def customer_aggregates(df):
    """Compute RFM aggregates per customer."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)
    reference_date = pd.Timestamp.now()
    customer_data = df.groupby('CustomerId').agg(
        recency_days=('TransactionStartTime', lambda x: (reference_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', lambda x: x[x > 0].sum())
    ).reset_index()
    print(customer_data.head())
    print(customer_data.describe())
    return customer_data

add eda__utils.py with reusable functions
