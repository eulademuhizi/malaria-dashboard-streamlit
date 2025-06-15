import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Malaria Sectors Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimal spacing and dark mode metrics
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
    }
    .stMetric {
        background-color: #2b2b2b;
        border: 1px solid #444;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0;
        color: white;
    }
    .stMetric > div {
        color: white;
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: white !important;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        color: white;
    }
    .element-container {
        margin: 0 !important;
    }
    .stPlotlyChart {
        margin: 0 !important;
    }
    /* Force metric text to be white */
    [data-testid="metric-container"] {
        background-color: #2b2b2b;
        border: 1px solid #444;
        padding: 0.5rem;
        border-radius: 0.25rem;
        color: white;
    }
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    [data-testid="metric-container"] label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the malaria sector data with performance optimizations"""
    try:
        # Load the GeoJSON file
        sector_data = gpd.read_file('sector_malaria.geojson')
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['Simple malaria cases', 'incidence', 'Population', 'year']
        for col in numeric_columns:
            sector_data[col] = pd.to_numeric(sector_data[col], errors='coerce').fillna(0)
        
        # Create unique sector names for disambiguation
        sector_data['sector_display'] = sector_data.apply(
            lambda row: f"{row['Sector']} ({row['District']})", axis=1
        )
        
        # Pre-calculate some commonly used aggregations for performance
        sector_data['sector_key'] = sector_data['Sector'] + '_' + sector_data['District']
        
        # Precompute sector options here to avoid hashing issues later
        sector_options = sector_data[['Sector', 'District', 'sector_display', 'sector_key']].drop_duplicates()
        sector_options = sorted(sector_options['sector_display'].tolist())
        
        return sector_data, sector_options  # Return both data and options
    except FileNotFoundError:
        st.error("sector_malaria.geojson file not found. Please ensure the file is in the correct directory.")
        return None, []
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, []

@st.cache_data
def calculate_metrics(_data, selected_year, selected_metric, previous_year=None):
    """Calculate key metrics for the dashboard - cached for performance"""
    current_data = _data[_data['year'] == selected_year]
    
    if selected_metric == 'Simple malaria cases':
        total_cases = current_data[selected_metric].sum()
        total_population = current_data['Population'].sum()
        # Calculate overall incidence: (total cases / total population) * 1000
        overall_incidence = (total_cases / total_population * 1000) if total_population > 0 else 0
    else:  # incidence
        # For incidence metric, take the mean of sector-level incidences
        overall_incidence = current_data[selected_metric].mean()
        total_cases = current_data['Simple malaria cases'].sum()
    
    change_percent = None
    if previous_year and previous_year in _data['year'].values:
        prev_data = _data[_data['year'] == previous_year]
        if selected_metric == 'Simple malaria cases':
            prev_total_cases = prev_data[selected_metric].sum()
            prev_total_pop = prev_data['Population'].sum()
            prev_incidence = (prev_total_cases / prev_total_pop * 1000) if prev_total_pop > 0 else 0
        else:  # incidence
            prev_incidence = prev_data[selected_metric].mean()
        
        if prev_incidence > 0:
            change_percent = ((overall_incidence - prev_incidence) / prev_incidence) * 100
    
    return total_cases, overall_incidence, change_percent

@st.cache_data
def get_color_scale_range(_data, metric):
    """Get the global min and max for consistent color scaling across years - cached"""
    return _data[metric].min(), _data[metric].max()

def create_choropleth_map(data, year, metric='Simple malaria cases'):
    """Create choropleth map using Plotly"""
    filtered_data = data[data['year'] == year].copy()
    
    # Get global range for consistent coloring
    vmin, vmax = get_color_scale_range(data, metric)
    
    if metric == 'Simple malaria cases':
        color_column = 'Simple malaria cases'
        title = f'Simple Malaria Cases - {year}'
        colorbar_title = 'Simple Malaria Cases'
    else:  # incidence
        color_column = 'incidence'
        title = f'Malaria Incidence - {year}'
        colorbar_title = 'Incidence'
    
    # Create custom pink to purple color scale
    pink_purple_scale = [
        [0.0, '#fce4ec'],    # Very light pink
        [0.2, '#f8bbd9'],    # Light pink
        [0.4, '#e91e63'],    # Medium pink
        [0.6, '#ad1457'],    # Dark pink
        [0.8, '#7b1fa2'],    # Light purple
        [1.0, '#4a148c']     # Deep purple
    ]
    
    # Create the map
    fig = px.choropleth_mapbox(
        filtered_data,
        geojson=filtered_data.geometry.__geo_interface__,
        locations=filtered_data.index,
        color=color_column,
        hover_name='sector_display',
        hover_data={
            'District': True,
            'Simple malaria cases': ':,.0f',
            'incidence': ':.2f'
        },
        color_continuous_scale=pink_purple_scale,
        range_color=[vmin, vmax],  # Set consistent color range
        mapbox_style='carto-positron',  # Light gray canvas
        zoom=6.8,
        center={'lat': -1.9, 'lon': 29.9},
        title=title,
        labels={
            'Simple malaria cases': 'Cases',
            'incidence': 'Incidence',
            'District': 'District'
        }
    )
    
    # Update layout for dark mode
    fig.update_layout(
        plot_bgcolor='rgba(20,20,20,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=14,
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(font=dict(color='white')),
        coloraxis_colorbar=dict(
            title_font_color='white',
            tickfont_color='white'
        )
    )
    
    return fig

def create_top_sectors_chart(data, year, metric='Simple malaria cases', top_n=10):
    """Create top sectors bar chart"""
    filtered_data = data[data['year'] == year].copy()
    
    sorted_data = filtered_data.nlargest(top_n, metric)
    
    if metric == 'Simple malaria cases':
        y_title = 'Simple Malaria Cases'
        title = f'Top {top_n} Simple Malaria Cases - {year}'
    else:  # incidence
        y_title = 'Incidence'
        title = f'Top {top_n} Malaria Incidence - {year}'
    
    # Pink to purple color scale
    pink_purple_scale = ['#fce4ec', '#f8bbd9', '#e91e63', '#ad1457', '#7b1fa2', '#4a148c']
    
    fig = px.bar(
        sorted_data,
        x=metric,
        y='sector_display',
        orientation='h',
        color=metric,
        color_continuous_scale=pink_purple_scale,
        title=title,
        labels={metric: y_title},
        hover_data={
            'District': True,
            'Simple malaria cases': ':,.0f',
            'incidence': ':.2f'
        }
    )
    
    # Update layout for dark mode
    fig.update_layout(
        plot_bgcolor='rgba(20,20,20,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=520,
        showlegend=False,
        title_font_size=14,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(font=dict(color='white')),
        xaxis=dict(
            title_font_color='white',
            tickfont_color='white',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            categoryorder='total ascending',
            title_font_color='white',
            tickfont_color='white',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        ),
        coloraxis_colorbar=dict(
            title_font_color='white',
            tickfont_color='white'
        )
    )
    
    return fig

def create_trend_chart(data, selected_sectors, metric='incidence'):
    """Create trend line chart for selected sectors with adjusted height"""
    if not selected_sectors:
        return None
    
    # Convert display names back to sector keys for filtering
    sector_keys = []
    for display_name in selected_sectors:
        matching_row = data[data['sector_display'] == display_name].iloc[0]
        sector_keys.append(matching_row['sector_key'])
    
    filtered_data = data[data['sector_key'].isin(sector_keys)]
    
    if metric == 'Simple malaria cases':
        y_column = 'Simple malaria cases'
        y_title = 'Simple Malaria Cases'
        title = 'Simple Malaria Cases Trends'
    else:  # incidence
        y_column = 'incidence'
        y_title = 'Incidence'
        title = 'Malaria Incidence Trends'
    
    # Define harmonized colors (blues and teals)
    harmonized_colors = [
        '#1f77b4',  # Blue
        '#17becf',  # Teal
        '#2ca02c',  # Green
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#d62728',  # Red
    ]
    
    # Create color mapping for consistency
    color_map = {display_name: harmonized_colors[i % len(harmonized_colors)] 
                 for i, display_name in enumerate(selected_sectors)}
    
    fig = px.line(
        filtered_data,
        x='year',
        y=y_column,
        color='sector_display',
        title=title,
        labels={y_column: y_title, 'year': 'Year'},
        color_discrete_map=color_map,
        hover_data={'District': True}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450,  # Match province plot height
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            font=dict(
                color='white',
                size=10
            ),
            bgcolor='rgba(30,30,30,0.9)',
            bordercolor='white',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=False,
            color='white'
        ),
        yaxis=dict(
            showgrid=False,
            color='white'
        ),
        title=dict(
            font=dict(color='white')
        )
    )
    
    # Enhanced line styling
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        mode='lines+markers'
    )
    
    return fig

def create_comparison_bar_chart(data, year, selected_sectors, metric):
    """Create bar chart comparing selected sectors for the given year and metric"""
    # Filter data for selected year and sectors
    filtered_data = data[data['year'] == year].copy()
    filtered_data = filtered_data[filtered_data['sector_display'].isin(selected_sectors)]
    
    if filtered_data.empty:
        return None
    
    # Sort by metric value
    filtered_data = filtered_data.sort_values(metric, ascending=False)
    
    # Create harmonized colors (same as trend chart)
    harmonized_colors = [
        '#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#d62728'
    ]
    
    # Create color mapping
    color_map = {sector: harmonized_colors[i % len(harmonized_colors)] 
                 for i, sector in enumerate(selected_sectors)}
    
    # Create the bar chart
    fig = px.bar(
        filtered_data,
        x='sector_display',
        y=metric,
        color='sector_display',
        color_discrete_map=color_map,
        labels={
            'sector_display': 'Sector',
            metric: metric.replace('_', ' ').title()
        },
        title=f"{metric.replace('_', ' ').title()} by Sector - {year}"
    )
    
    # Update layout for dark mode
    fig.update_layout(
        plot_bgcolor='rgba(20,20,20,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        showlegend=False,
        xaxis_title="Sector",
        yaxis_title=metric.replace('_', ' ').title(),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(font=dict(color='white')),
        xaxis=dict(
            title_font_color='white',
            tickfont_color='white',
            showgrid=False
        ),
        yaxis=dict(
            title_font_color='white',
            tickfont_color='white',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_district_heatmap(data, metric='incidence'):
    """Create district-level heatmap showing trends over time"""
    
    # Group data by district and year
    district_data = data.groupby(['District', 'year']).agg({
        'Simple malaria cases': 'sum',
        'incidence': 'mean',
        'Population': 'sum'
    }).reset_index()
    
    # Calculate overall incidence for districts (if using cases metric)
    if metric == 'Simple malaria cases':
        district_data['calculated_incidence'] = (
            district_data['Simple malaria cases'] / district_data['Population'] * 1000
        ).fillna(0)
        value_column = 'calculated_incidence'
        title = 'District Malaria Incidence Over Time'
        colorbar_title = 'Incidence Rate'
    else:
        value_column = 'incidence'
        title = 'District Malaria Incidence Over Time'
        colorbar_title = 'Incidence Rate'
    
    # Create pivot table for heatmap
    heatmap_data = district_data.pivot(
        index='District', 
        columns='year', 
        values=value_column
    ).fillna(0)
    
    # Sort districts by overall incidence (descending)
    district_means = heatmap_data.mean(axis=1).sort_values(ascending=False)
    heatmap_data = heatmap_data.reindex(district_means.index)
    
    # Create custom color scale (white to red)
    custom_colorscale = [
        [0.0, '#ffffff'],    # White
        [0.1, '#ffe6e6'],    # Very light pink
        [0.2, '#ffcccc'],    # Light pink
        [0.3, '#ff9999'],    # Pink
        [0.4, '#ff6666'],    # Medium pink
        [0.5, '#ff3333'],    # Red-pink
        [0.6, '#ff0000'],    # Red
        [0.7, '#cc0000'],    # Dark red
        [0.8, '#990000'],    # Darker red
        [0.9, '#660000'],    # Very dark red
        [1.0, '#330000']     # Almost black red
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(
            title=colorbar_title,
            titleside='right',
            tickmode='linear',
            dtick=10
        ),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>' +
                      'Year: %{x}<br>' +
                      colorbar_title + ': %{z:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        title_font_size=14,
        xaxis_title='Year',
        yaxis_title='District',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        xaxis=dict(
            showgrid=False,
            tickmode='linear',
            dtick=1,
            title_font_size=12
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10),
            title_font_size=12
        )
    )
    
    return fig

def create_province_scatterplot(data, year=2025):
    """Create scatterplot showing Population vs Incidence with Province as hue and quadrant lines - ALWAYS uses 2025 data"""
    # FIXED: Always filter data for 2025 regardless of the year parameter
    filtered_data = data[data['year'] == 2025].copy()
    
    # Only remove rows with zero or negative population
    filtered_data = filtered_data[filtered_data['Population'] > 0].copy()
    
    if filtered_data.empty:
        return None, None, None
    
    # Calculate robust thresholds using percentiles
    pop_threshold = np.percentile(filtered_data['Population'], 75)  # 75th percentile for population
    inc_threshold = np.percentile(filtered_data['incidence'], 75)  # 75th percentile for incidence
    
    # Calculate y-axis limits based on data distribution
    max_incidence = filtered_data['incidence'].max()
    # Set y-axis upper bound to 120% of max incidence or 1.5x threshold, whichever is larger
    y_upper_bound = max(max_incidence * 1.2, inc_threshold * 1.5)
    
    # Create the scatterplot - title shows it's always 2025
    fig = px.scatter(
        filtered_data,
        x='Population',
        y='incidence',
        color='Province',
        hover_name='sector_display',
        hover_data={
            'District': True,
            'incidence': ':.2f',
            'Simple malaria cases': ':,.0f',
            'Province': False
        },
        title=f'Population vs Incidence by Province - 2025',  # FIXED: Always shows 2025
        labels={
            'Population': 'Population',
            'incidence': 'Incidence'
        },
        opacity=0.8
    )
    
    # Add quadrant lines that span the entire plot
    fig.add_shape(
        type="line",
        x0=pop_threshold, 
        y0=0,
        x1=pop_threshold, 
        y1=y_upper_bound,
        line=dict(color="white", width=2, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=filtered_data['Population'].min() * 0.9, 
        y0=inc_threshold,
        x1=filtered_data['Population'].max() * 1.1, 
        y1=inc_threshold,
        line=dict(color="white", width=2, dash="dash")
    )
    
    # Calculate positions for quadrant labels
    y_upper = y_upper_bound * 0.9
    y_lower = inc_threshold * 0.2
    
    # Add quadrant annotations
    fig.add_annotation(
        x=pop_threshold/2, 
        y=y_upper,
        text="Low Pop, High Incidence<br>(Hotspots)",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=pop_threshold*2, 
        y=y_upper,
        text="High Pop, High Incidence<br>(Priority Zones)",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=pop_threshold/2, 
        y=y_lower,
        text="Low Pop, Low Incidence<br>(Stable Areas)",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    fig.add_annotation(
        x=pop_threshold*2, 
        y=y_lower,
        text="High Pop, Low Incidence<br>(Protected Areas)",
        showarrow=False,
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    # Update layout for dark mode with optimized scaling
    fig.update_layout(
        plot_bgcolor='rgba(20,20,20,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450,  # Reduced height to match trend chart
        margin=dict(l=0, r=0, t=40, b=0),
        title_font_size=14,
        title=dict(font=dict(color='white')),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            title_font_size=12,
            color='white',
            type='log',
            range=[np.log10(filtered_data['Population'].min() * 0.9), 
                   np.log10(filtered_data['Population'].max() * 1.1)]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.2)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            title_font_size=12,
            color='white',
            range=[0, y_upper_bound]  # Set dynamic y-axis range
        ),
        legend=dict(
            title='Province',
            font=dict(size=10, color='white'),
            bgcolor='rgba(30,30,30,0.9)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    # Update scatter points for better visibility
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='white')),
        selector=dict(mode='markers')
    )
    
    return fig, pop_threshold, inc_threshold

def main():
    # Title
    st.title("Rwanda Malaria Sectors Dashboard")
    
    # Load data
    data, sector_options = load_data()
    if data is None:
        st.stop()
    
    # Correct province name
    data['Province'] = data['Province'].replace('Iburengerazuba', 'Kigali')
    data['Province'] = data['Province'].replace('Kigali', 'Kigali City')
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    years = sorted(data['year'].unique())
    # Slider for year selection
    selected_year = st.sidebar.slider("Year", min_value=min(years), max_value=max(years), value=max(years), step=1)
    
    metric_options = {
        'Simple Malaria Cases': 'Simple malaria cases',
        'Incidence': 'incidence'
    }
    selected_metric_display = st.sidebar.selectbox("Primary Metric", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_display]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sector Analysis")
    
    # Direct sector selection with limit
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors for Comparison (Max 10)", 
        sector_options, 
        default=[],
        help="Select up to 10 sectors to analyze trends. Format: Sector (District)",
        max_selections=10
    )
    
    # Calculate metrics
    previous_year = selected_year - 1 if selected_year > min(years) else None
    total_cases, overall_incidence, change_percent = calculate_metrics(data, selected_year, selected_metric, previous_year)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Simple Malaria Cases", f"{total_cases:,.0f}")
    
    with col2:
        st.metric("Average Incidence", f"{overall_incidence:.2f}")
    
    with col3:
        if change_percent is not None:
            if change_percent > 0:
                arrow = "↑"
                color_style = "color: red;"
            else:
                arrow = "↓"
                color_style = "color: green;"
                change_percent = abs(change_percent)
            
            change_html = f"""
            <div style="background-color: #2b2b2b; border: 1px solid #444; padding: 0.5rem; border-radius: 0.25rem;">
                <div style="font-size: 14px; color: white;">Change(%) of incidence</div>
                <div style="font-size: 28px; font-weight: 600; {color_style}">
                    {arrow} {change_percent:+.1f}% vs {previous_year}
                </div>
            </div>
            """
            st.markdown(change_html, unsafe_allow_html=True)
        else:
            st.metric("Change(%) of incidence", "N/A")
    
    # Map and Top Sectors
    map_col, chart_col = st.columns([2.5, 1])
    
    with map_col:
        map_fig = create_choropleth_map(data, selected_year, selected_metric)
        st.plotly_chart(map_fig, use_container_width=True)
    
    with chart_col:
        top_sectors_fig = create_top_sectors_chart(data, selected_year, selected_metric)
        st.plotly_chart(top_sectors_fig, use_container_width=True)
    
    # NEW LAYOUT: Two columns under the map
    st.subheader("Sector Analysis Dashboard")
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Sector Trends
        st.markdown("### Sector Trends Over Time")
        if selected_sectors:
            trend_fig = create_trend_chart(data, selected_sectors, selected_metric)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Select sectors from the sidebar to view trends")
            st.markdown("""
            **How to Use:**
            1. Select sectors in the sidebar
            2. Compare trends over time
            3. Identify patterns:
               - Consistent improvement
               - Recent deterioration
               - Seasonal patterns
            """)
    
    with col_right:
        # Province Analysis (now on the right) - FIXED: Always shows 2025 data
        st.markdown("### Province Analysis")
        province_fig, pop_threshold, inc_threshold = create_province_scatterplot(data, year=2025)  # FIXED: Always pass 2025
        if province_fig:
            st.plotly_chart(province_fig, use_container_width=True)
            
            # Add the requested quadrant comments
            st.markdown("""
            **Analysis Guide:**

            🟥 **Upper-right quadrant**: High incidence & high population – major burden areas requiring scaled interventions.
            *Interventions: mass bed net distribution, regular indoor spraying, expanded rapid testing and treatment at health facilities and community level*

            🟧 **Upper-left quadrant**: High incidence & low population – hotspot zones needing focused action.
            *Interventions: targeted spraying and larviciding, fixed testing points in risk zones, active follow-up of cases by health workers*

            🟨 **Lower-right quadrant**: Low incidence & high population – controlled areas to maintain success.
            *Interventions: routine fever testing in clinics, malaria prevention in schools, periodic net replacement and monitoring*

            🟩 **Lower-left quadrant**: Low incidence & low population – low‑risk zones, monitor periodically.
            *Interventions: regular vector surveillance, repellent distribution, continued public awareness through local health networks*
            """)

if __name__ == "__main__":
    main()