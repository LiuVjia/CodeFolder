import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt # Removed Matplotlib
import numpy_financial as npf
import pandas as pd # Using pandas for better data table handling
import math
import plotly.graph_objects as go # Added Plotly
import plotly.express as px # Added Plotly

# Set page configuration
st.set_page_config(
    page_title="General Financial Cash Flow Analysis Tool",
    page_icon="💰",
    layout="wide"
)

# --- UI Elements and Input ---

# Add a title and an optional introductory image/emoji
st.title("📈 General Financial Cash Flow Analysis Tool (Customizable)")
# Example of adding an image (replace 'path/to/your/image.png' with your image file path)
# st.image('path/to/your/image.png', caption='Cash Flow Analysis')

st.markdown("""
    Welcome! This tool allows you to analyze the Net Present Value (NPV),
    Internal Rate of Return (IRR), and Payback Period for your project
    by defining custom initial investments, recurring cash inflows/outflows,
    and specific maintenance costs based on accumulated mileage.
""")

# Initialize session state variables if not already done
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    # Initialize custom cash flow items
    st.session_state.inflows = []
    st.session_state.outflows = []
    st.session_state.initial_investments = []
    # Initialize maintenance cost table with column names (will be filled with defaults below if empty)
    st.session_state.maintenance_cost_table = pd.DataFrame({
        "Accumulated Mileage (km)": pd.Series(dtype='int'),
        "Cost Per Unit (£)": pd.Series(dtype='float')
    })


# 1. Project Basic Information
st.header("1. Project Basic Information")
col1, col2 = st.columns(2)
with col1:
    project_period_years = st.number_input("Project Period (Years)", min_value=1, max_value=50, value=5)
with col2:
    discount_rate = st.number_input("Discount Rate / Cost of Capital (%)", value=9.0) / 100

# 2. Initial Investments Setup (Can add multiple items)
st.header("2. Initial Investments Setup 💰")
st.markdown("Add all initial investment items (typically negative cash flows):")
default_initial_investments = [
    {"Item Name": "Total Initial Investment", "Amount (£)": -2000000} # Example, can be modified or add more
]
st.session_state.initial_investments = st.data_editor(
    st.session_state.initial_investments if st.session_state.initial_investments else default_initial_investments,
    num_rows="dynamic",
    key="initial_investments_editor",
    use_container_width=True
)

total_initial_investment = sum([item.get("Amount (£)", 0) for item in st.session_state.initial_investments])
st.write(f"**Total Initial Investment (Year 0): £{total_initial_investment:,.0f}**")


# 3. Recurring Cash Flow Setup (Customizable Items)
st.header("3. Recurring Cash Flow Setup 🔄")
st.markdown("Add annual cash inflows (savings/revenue) and cash outflows (operating costs). You can set annual growth and the starting year for growth.")

tab_inflows, tab_outflows, tab_maintenance = st.tabs(["Cash Inflows", "Cash Outflows", "Maintenance Costs (by Mileage)"])

with tab_inflows:
    st.markdown("Add recurring annual cash inflow items (enter positive values):")
    default_inflows = [
        {"Item Name": "Outsourcing Savings", "Base Amount (£)": 3000000, "Consider Annual Growth": True, "Annual Growth Rate (%)": 2.0, "Growth Start Year": 3}
    ]
    st.session_state.inflows = st.data_editor(
        st.session_state.inflows if st.session_state.inflows else default_inflows,
        num_rows="dynamic",
        key="inflows_editor",
        use_container_width=True,
        column_config={
            "Consider Annual Growth": st.column_config.CheckboxColumn(required=True),
            "Annual Growth Rate (%)": st.column_config.NumberColumn(format="%.2f", default=0.0),
            "Growth Start Year": st.column_config.NumberColumn(min_value=1, default=1)
        }
    )

with tab_outflows:
    st.markdown("Add recurring annual cash outflow items (enter positive values):")
    # Note: Example calculations based on original code structure, adjust as needed
    # Assuming unit_count was a parameter for calculating these base costs in original code
    # For a truly general tool, these base amounts would be direct inputs
    # Here we keep the structure for demonstration but ideally, remove dependencies on unit_count etc.
    # Example with calculated base amounts:
    # unit_count_example = 40
    # annual_mileage_example = 40000
    # fuel_economy_example = 16.0
    # fuel_price_example = 1.45
    # salary_per_driver_example = 20000
    # drivers_per_vehicle_example = 2
    # garage_per_vehicle_example = 1500
    # insurance_per_vehicle_example = 1200
    # general_inflation_rate_example = 2.0

    default_outflows = [
        {"Item Name": "Driver Salaries", "Base Amount (£)": 20000 * 2 * 40, "Consider Annual Growth": True, "Annual Growth Rate (%)": 2.0, "Growth Start Year": 2},
        {"Item Name": "Fuel Costs", "Base Amount (£)": (40000 / 100) * 16.0 * 1.45 * 40, "Consider Annual Growth": True, "Annual Growth Rate (%)": 2.0, "Growth Start Year": 2},
        {"Item Name": "Garage Fees", "Base Amount (£)": 1500 * 40, "Consider Annual Growth": True, "Annual Growth Rate (%)": 2.0, "Growth Start Year": 2},
        {"Item Name": "Insurance Costs", "Base Amount (£)": 1200 * 40, "Consider Annual Growth": True, "Annual Growth Rate (%)": 2.0, "Growth Start Year": 2},
    ]
    st.session_state.outflows = st.data_editor(
        st.session_state.outflows if st.session_state.outflows else default_outflows,
        num_rows="dynamic",
        key="outflows_editor",
        use_container_width=True,
        column_config={
            "Consider Annual Growth": st.column_config.CheckboxColumn(required=True),
            "Annual Growth Rate (%)": st.column_config.NumberColumn(format="%.2f", default=0.0),
            "Growth Start Year": st.column_config.NumberColumn(min_value=1, default=1)
        }
    )

with tab_maintenance:
     st.markdown("**Maintenance Cost Table (Per Accumulated Mileage Per Unit)**")
     st.markdown("Define the cost incurred per unit when reaching specific accumulated mileage points.")

     # Define default maintenance table data
     default_maintenance_table_data = [
         {"Accumulated Mileage (km)": 20000, "Cost Per Unit (£)": 600},
         {"Accumulated Mileage (km)": 40000, "Cost Per Unit (£)": 1200},
         {"Accumulated Mileage (km)": 60000, "Cost Per Unit (£)": 600},
         {"Accumulated Mileage (km)": 80000, "Cost Per Unit (£)": 2000},
         {"Accumulated Mileage (km)": 100000, "Cost Per Unit (£)": 600},
         {"Accumulated Mileage (km)": 120000, "Cost Per Unit (£)": 1200},
         {"Accumulated Mileage (km)": 140000, "Cost Per Unit (£)": 600},
         {"Accumulated Mileage (km)": 160000, "Cost Per Unit (£)": 2000},
         {"Accumulated Mileage (km)": 180000, "Cost Per Unit (£)": 600},
         {"Accumulated Mileage (km)": 200000, "Cost Per Unit (£)": 1200},
     ]

     # Initialize maintenance cost table in session state if it\'s not exist or is empty
     if 'maintenance_cost_table' not in st.session_state or st.session_state.maintenance_cost_table.empty:
         # Initialize with default data as a DataFrame
         st.session_state.maintenance_cost_table = pd.DataFrame(default_maintenance_table_data)

     # Display and edit the maintenance cost table
     # Ensure the editor is always populated with the session state data
     maintenance_cost_table_data = st.data_editor(
         st.session_state.maintenance_cost_table,
         num_rows="dynamic",
         use_container_width=True
     )

     # Update session state with the edited data
     st.session_state.maintenance_cost_table = maintenance_cost_table_data

     st.markdown("Mileage information required for maintenance calculation:")
     col_m1, col_m2 = st.columns(2)
     with col_m1:
         annual_mileage_per_unit_maint = st.number_input("Annual Mileage Per Unit (km)", value=40000)
     with col_m2:
         units_for_maint_calc = st.number_input("Number of Units for Maintenance Calculation", min_value=0, value=40) # Units count for maintenance costs


# 4. Terminal Cash Flow Setup (e.g., Salvage Value)
st.header("4. Terminal Cash Flow Setup 🔚")
st.markdown("Cash flows occurring in the final year of the project (e.g., asset salvage value).")
# This section can be expanded similarly to recurring cash flows if needed
col_term1, col_term2 = st.columns(2)
with col_term1:
    salvage_value_per_unit = st.number_input("Salvage Value Per Unit (£, Final Year)", value=2000)
with col_term2:
    units_for_salvage = st.number_input("Number of Units for Salvage Value Calculation", min_value=0, value=40 if project_period_years >= 1 else 0) # Only applicable if period >= 1 year
total_salvage_value = salvage_value_per_unit * units_for_salvage if project_period_years >= 1 else 0


# --- Cash Flow Calculation Logic ---

# Helper function to calculate total maintenance cost per unit based on accumulated mileage
def get_total_maintenance_cost_per_unit(accumulated_mileage, maintenance_table_data):
    """Calculates the total maintenance cost for a single unit based on accumulated mileage
       by summing up costs at each triggered mileage point in the table."""
    total_fee = 0
    # Ensure maintenance_table_data is a valid DataFrame and not empty
    if isinstance(maintenance_table_data, pd.DataFrame) and not maintenance_table_data.empty:
        # Sort the table by mileage to ensure correct calculation for triggered points
        # Use .copy() to avoid SettingWithCopyWarning if we were modifying the DataFrame
        sorted_table = maintenance_table_data.sort_values(by="Accumulated Mileage (km)").copy()

        # Sum up costs for all mileage points reached
        for index, row in sorted_table.iterrows():
            # Use .get() with a default for safety, though column names should be guaranteed
            mileage_point = row.get("Accumulated Mileage (km)", 0)
            cost_at_point = row.get("Cost Per Unit (£)", 0)
            if accumulated_mileage >= mileage_point:
                total_fee += cost_at_point # Cost is incurred when this mileage point is reached

    return total_fee

annual_cash_flows_list = [] # Cash flows from Year 1 to Year period
annual_detail_display_list = [] # For displaying detailed annual breakdown (text format)
annual_breakdown_table_data = [] # Data for the detailed breakdown table

# Prepare column names for the detailed breakdown table based on outflow items
# Ensure unique column names in case item names are not unique
outflow_item_base_names = [item.get("Item Name", f"Outflow {i+1}") for i, item in enumerate(st.session_state.outflows)]
# Create a dictionary to count occurrences of item names
name_counts = {}
outflow_item_column_names = []
for name in outflow_item_base_names:
    if name in name_counts:
        name_counts[name] += 1
        outflow_item_column_names.append(f"{name} {name_counts[name]} (£)")
    else:
        name_counts[name] = 1
        outflow_item_column_names.append(f"{name} (£)")


breakdown_table_columns = ["Year", "Total Inflows (£)"] + outflow_item_column_names + ["Maintenance Costs (£)", "Terminal Cash Flow (Salvage) (£)", "Net Cash Flow (£)", "Accumulated Mileage (km)"]


# Year 0 Cash Flow (Total Initial Investment)
# irr_cash_flows will include Year 0 for NPV/IRR calculations
irr_cash_flows = [total_initial_investment]
annual_detail_display_list.append({
    "Year": 0,
    "Description": "Initial Investment",
    "Cash Inflow": 0,
    "Cash Outflow": -total_initial_investment, # Display as positive for outflow column
    "Net Cash Flow": total_initial_investment # This is the actual cash flow value for Year 0
})
# Prepare Year 0 data for the detailed breakdown table
year_0_table_row = {"Year": 0, "Total Inflows (£)": 0, "Maintenance Costs (£)": 0, "Terminal Cash Flow (Salvage) (£)": 0, "Net Cash Flow (£)": total_initial_investment, "Accumulated Mileage (km)": 0}
for col_name in outflow_item_column_names:
    year_0_table_row[col_name] = 0 # Initialize outflow columns to 0 for Year 0

annual_breakdown_table_data.append(year_0_table_row)


# Calculate Cash Flows for Year 1 to Year period
for year in range(1, int(project_period_years) + 1):
    cash_in_this_year = 0
    cash_out_this_year_excl_maint = 0 # Outflows excluding maintenance
    maintenance_cost_this_year = 0
    terminal_cash_flow_this_year = 0

    operating_cost_detail = {} # Detailed breakdown for text display
    inflow_detail_this_year = {} # Detailed breakdown for text display
    outflow_detail_this_year_excl_maint = {} # Detailed breakdown for text display

    # Dictionary to hold individual outflow amounts for the table row
    individual_outflow_amounts = {}
    # Initialize individual outflow amounts for the current year
    for item_name_col in outflow_item_column_names:
        individual_outflow_amounts[item_name_col] = 0


    # Calculate Recurring Cash Inflows for the year
    for inflow_item in st.session_state.inflows:
        item_name = inflow_item.get("Item Name", "Unnamed Inflow")
        base_amount = inflow_item.get("Base Amount (£)", 0)
        consider_growth = inflow_item.get("Consider Annual Growth", False)
        growth_rate = inflow_item.get("Annual Growth Rate (%)", 0.0) / 100
        growth_start_year = inflow_item.get("Growth Start Year", 1)

        current_inflow = base_amount
        if consider_growth and year >= growth_start_year:
             # Growth factor starts applying from the growth start year
             growth_factor = (1 + growth_rate) ** (year - growth_start_year + 1)
             current_inflow = base_amount * growth_factor

        cash_in_this_year += current_inflow
        inflow_detail_this_year[item_name] = current_inflow

    # Calculate Recurring Cash Outflows for the year (excluding maintenance)
    # Need to handle potential duplicate item names when assigning to individual_outflow_amounts
    name_counts_this_year = {} # Counter for this year's item names to handle duplicates

    for outflow_item in st.session_state.outflows:
        item_name_base = outflow_item.get("Item Name", "Unnamed Outflow")
        base_amount = outflow_item.get("Base Amount (£)", 0)
        consider_growth = outflow_item.get("Consider Annual Growth", False)
        growth_rate = outflow_item.get("Annual Growth Rate (%)", 0.0) / 100
        growth_start_year = outflow_item.get("Growth Start Year", 1)

        current_outflow = base_amount
        if consider_growth and year >= growth_start_year:
             # Growth factor starts applying from the growth start year
             growth_factor = (1 + growth_rate) ** (year - growth_start_year + 1)
             current_outflow = base_amount * growth_factor

        cash_out_this_year_excl_maint += current_outflow
        outflow_detail_this_year_excl_maint[item_name_base] = current_outflow # Use base name for detail text

        # Determine the correct column name for this item, handling duplicates
        if item_name_base in name_counts_this_year:
            name_counts_this_year[item_name_base] += 1
            item_name_col = f"{item_name_base} {name_counts_this_year[item_name_base]} (£)"
        else:
            name_counts_this_year[item_name_base] = 1
            item_name_col = f"{item_name_base} (£)"

        individual_outflow_amounts[item_name_col] = current_outflow # Store individual outflow amount


    # Calculate Maintenance Costs for the year (by accumulated mileage)
    accumulated_mileage_this_year = annual_mileage_per_unit_maint * year
    previous_year_accumulated_mileage = annual_mileage_per_unit_maint * (year - 1)

    # Calculate the *additional* maintenance cost incurred this year per unit
    total_maint_per_unit_this_year = get_total_maintenance_cost_per_unit(accumulated_mileage_this_year, st.session_state.maintenance_cost_table)
    total_maint_per_unit_previous_year = get_total_maintenance_cost_per_unit(previous_year_accumulated_mileage, st.session_state.maintenance_cost_table)
    maintenance_cost_base_this_year_per_unit = total_maint_per_unit_this_year - total_maint_per_unit_previous_year

    # Total maintenance cost for the year across all units
    maintenance_cost_this_year_before_inflation = maintenance_cost_base_this_year_per_unit * units_for_maint_calc

    # Decide if maintenance cost is subject to general inflation.
    # Based on previous logic, it seemed to be. We'll add an option or clarify.
    # For now, let's assume it follows the first recurring outflow's growth setting if that exists and applies from year 2.
    general_inflation_rate = 0.0
    general_inflation_start_year = 1 # Default to start from year 1
    consider_general_inflation = False
    # Find a suitable outflow to derive a general inflation rate from, if any
    first_applicable_outflow = next((item for item in st.session_state.outflows if item.get("Consider Annual Growth", False)), None)

    if first_applicable_outflow:
        general_inflation_rate = first_applicable_outflow.get("Annual Growth Rate (%)", 0.0) / 100
        general_inflation_start_year = first_applicable_outflow.get("Growth Start Year", 1)
        consider_general_inflation = True


    # Apply general inflation to maintenance cost if applicable and from Year 2 onwards (based on original logic)
    # Adjusted growth exponent and start year for clarity
    inflation_adjusted_maintenance_cost_this_year = maintenance_cost_this_year_before_inflation
    if consider_general_inflation and year >= max(2, general_inflation_start_year):
         # Calculate years *since* the inflation started for maintenance (Year 2 or growth start year, whichever is later)
         years_since_inflation_start = year - max(2, general_inflation_start_year) + 1
         if years_since_inflation_start > 0: # Only apply growth if years_since_inflation_start is positive
            inflation_adjusted_maintenance_cost_this_year *= ((1 + general_inflation_rate) ** years_since_inflation_start)

    maintenance_cost_this_year = inflation_adjusted_maintenance_cost_this_year # Update maintenance cost with inflation


    # Calculate Terminal Cash Flow (Salvage value occurs only in the final year)
    if year == int(project_period_years):
        terminal_cash_flow_this_year = total_salvage_value


    # Calculate Net Cash Flow for the year
    # Net Cash Flow = Total Inflows - (Sum of all Outflows including Maintenance) + Terminal Cash Flow
    total_outflows_incl_maint = sum(individual_outflow_amounts.values()) + maintenance_cost_this_year
    net_cash_flow_this_year = cash_in_this_year - total_outflows_incl_maint + terminal_cash_flow_this_year


    annual_cash_flows_list.append(net_cash_flow_this_year)
    irr_cash_flows.append(net_cash_flow_this_year) # Add to the sequence including Year 0

    # Store detailed information for text display
    # operating_cost_detail already contains non-maintenance outflows
    operating_cost_detail["Maintenance Costs"] = maintenance_cost_this_year # Add maintenance to combined outflows

    annual_detail_display_list.append({
        "Year": year,
        "Cash Inflows Detail": inflow_detail_this_year,
        "Cash Outflows Detail": operating_cost_detail, # Includes all outflows for text display
        "Terminal Cash Flow (Salvage)": terminal_cash_flow_this_year,
        "Net Cash Flow (£)": net_cash_flow_this_year,
        "Accumulated Mileage (km)": accumulated_mileage_this_year
    })

    # Add data to the detailed breakdown table data
    year_breakdown_row = {
        "Year": year,
        "Total Inflows (£)": cash_in_this_year,
        "Maintenance Costs (£)": maintenance_cost_this_year,
        "Terminal Cash Flow (Salvage) (£)": terminal_cash_flow_this_year,
        "Net Cash Flow (£)": net_cash_flow_this_year,
        "Accumulated Mileage (km)": accumulated_mileage_this_year
    }
    # Add individual outflow amounts to the row
    year_breakdown_row.update(individual_outflow_amounts)

    annual_breakdown_table_data.append(year_breakdown_row)


# --- Calculation Results and Display ---

st.header("5. Analysis Results ✨")

st.subheader("Annual Cash Flow Breakdown")
st.markdown("Detailed breakdown of cash inflows, outflows, and net cash flow for each year.")

# Display formulas
st.markdown(r"**Formulas:**")
st.markdown(r"Total Inflows = Sum of all Inflow Items for the year (with growth)")
# Updated formula description for total outflows (sum of individual items + maintenance)
outflow_formula_parts = [name.replace(' (£)', '') for name in outflow_item_column_names] + ["Maintenance Costs"]
outflow_formula_desc = " + ".join(outflow_formula_parts)
st.markdown(f"Total Outflows (incl. Maint.) = {outflow_formula_desc}")
st.markdown(r"Maintenance Costs = (Total Maintenance Cost per Unit for current year\'s Accumulated Mileage - Total Maintenance Cost per Unit for previous year\'s Accumulated Mileage) * Number of Units for Maintenance Calculation * (1 + General Inflation Rate)^(Years since Inflation Start for Maintenance)")
st.markdown(r"Terminal Cash Flow (Salvage) = Salvage Value Per Unit * Number of Units for Salvage Value Calculation (only in final year)")
st.markdown(r"**Net Cash Flow = Total Inflows - Total Outflows (incl. Maint.) + Terminal Cash Flow**")


# Display the annual breakdown table
annual_breakdown_df = pd.DataFrame(annual_breakdown_table_data)
# Ensure column order is correct for the DataFrame
ordered_columns = ["Year", "Total Inflows (£)"] + outflow_item_column_names + ["Maintenance Costs (£)", "Terminal Cash Flow (Salvage) (£)", "Net Cash Flow (£)", "Accumulated Mileage (km)"]
# Reindex to ensure all expected columns are present, filling missing ones with 0 (important for Year 0 where outflows are 0)
annual_breakdown_df = annual_breakdown_df.reindex(columns=ordered_columns, fill_value=0).set_index("Year")

st.dataframe(annual_breakdown_df, use_container_width=True)


st.subheader("Detailed Itemized Breakdown (Supplemental)")
st.markdown("Specific amounts for each individual inflow and outflow item (text format).")

# Keep the text display for detailed itemized breakdown as supplemental information
# Display details for all years (text format)
for detail in annual_detail_display_list: # Iterate through all years, including Year 0
    if detail['Year'] == 0:
         st.write(
            f"**Year {detail['Year']}:** "
            f"Description: {detail['Description']}"
            f" = **Net Cash Flow £{detail['Net Cash Flow']:,.0f}**"
         )
    else:
        st.write(
            f"**Year {detail['Year']}:** "
            + f"Total Inflows £{sum(detail['Cash Inflows Detail'].values()):,.0f}"
            # Summing all outflows including maintenance for this summary line in text display
            + f" - Total Outflows £{sum(detail['Cash Outflows Detail'].values()):,.0f}"
            + (f" + Terminal CF (Salvage) £{detail['Terminal Cash Flow (Salvage)']:,.0f}" if detail['Terminal Cash Flow (Salvage)'] else "")
            + f" = **Net Cash Flow £{detail['Net Cash Flow (£)']:,.0f}**" # Corrected key
        )
        # Display inflow details
        if detail['Cash Inflows Detail']:
             inflow_str = ', '.join([f'{k}: £{v:,.0f}' for k, v in detail['Cash Inflows Detail'].items()])
             st.caption(f"Inflows Detail: {inflow_str}")
        # Display outflow details (including maintenance)
        if detail['Cash Outflows Detail']:
             outflow_str = ', '.join([f'{k}: £{v:,.0f}' for k, v in detail['Cash Outflows Detail'].items()])
             st.caption(f"Outflows Detail: {outflow_str}")
        st.caption(f"Accumulated Mileage: {detail['Accumulated Mileage (km)']:,} km")


st.subheader("Net Present Value (NPV) Calculation Process")
st.markdown(f"**Formula:**")
st.markdown(r"$$ NPV = \sum_{n=0}^{Period} \frac{CF_n}{(1+r)^n} $$")
st.markdown(r"Where \(CF_n\) is the net cash flow in year \(n\), and \(r\) is the discount rate.")

npv_calculation_data = []
cumulative_npv_calc = 0

# Year 0 NPV calculation
year_0_cash_flow_calc = irr_cash_flows[0]
discount_factor_0 = 1.0
present_value_0 = year_0_cash_flow_calc * discount_factor_0
cumulative_npv_calc += present_value_0
npv_calculation_data.append({"Year": 0, "Cash Flow (£)": year_0_cash_flow_calc, "Discount Factor": discount_factor_0, "Present Value (£)": present_value_0, "Cumulative Present Value (£)": cumulative_npv_calc})

# Years 1 onwards NPV calculation
for year in range(1, int(project_period_years) + 1):
    # Get the net cash flow for this year from the sequence including Year 0
    cf = irr_cash_flows[year]
    discount_factor = 1 / ((1 + discount_rate) ** year)
    present_value = cf * discount_factor
    cumulative_npv_calc += present_value
    npv_calculation_data.append({"Year": year, "Cash Flow (£)": cf, "Discount Factor": discount_factor, "Present Value (£)": present_value, "Cumulative Present Value (£)": cumulative_npv_calc})

# Display NPV calculation table
npv_df = pd.DataFrame(npv_calculation_data).set_index("Year")
st.dataframe(npv_df, use_container_width=True) # Removed column_format

# Final NPV is the cumulative present value in the last year
final_npv = cumulative_npv_calc


st.subheader("Summary Investment Metrics")

st.markdown(f"**Net Present Value (NPV):** The difference between the present value of cash inflows and the present value of cash outflows over a period of time. A positive NPV generally indicates a profitable project.")
st.write(f"Net Present Value (NPV): £{final_npv:,.0f}")

st.markdown(f"**Payback Period:** The time required for the cumulative net cash flows from a project to equal the initial investment.")

# Payback Period Calculation
# Use the cumulative cash flows including Year 0 for payback period
cumulative_cash_flows_payback = np.cumsum(irr_cash_flows).tolist()

final_payback_period = None
# Check if cumulative cash flow ever turns positive
if any(cf >= 0 for cf in cumulative_cash_flows_payback):
    for i in range(len(cumulative_cash_flows_payback)):
        if cumulative_cash_flows_payback[i] >= 0:
            if i == 0:
                final_payback_period = 0.0
            else:
                # Payback occurs between year i-1 and i
                previous_cumulative_cf = cumulative_cash_flows_payback[i-1]
                current_year_cf = irr_cash_flows[i]

                if current_year_cf > 0:
                    # Calculate the fraction of the year needed to recover the remaining investment
                    fraction_of_year = abs(previous_cumulative_cf) / current_year_cf
                    final_payback_period = (i - 1) + fraction_of_year
                elif current_year_cf == 0 and previous_cumulative_cf >= 0:
                     final_payback_period = i - 1 # Payback at the end of previous year
                elif current_year_cf <= 0 and previous_cumulative_cf < 0:
                     # If CF is negative or zero, and cumulative was still negative, continue
                     continue
                else: # current_year_cf <= 0 and previous_cumulative_cf >= 0
                    # This case indicates payback happened before or at the start of this year, but CF is negative.
                    # Payback period is the end of the previous year.
                     final_payback_period = i -1 if i > 0 else 0
            break # Found the payback period


if final_payback_period is not None:
     st.write(f"Payback Period: {final_payback_period:.2f} Years")
else:
     st.write("Payback Period: Not Recovered within project period")


st.markdown(f"**Internal Rate of Return (IRR):** The discount rate at which the net present value (NPV) of a project equals zero. A higher IRR indicates a more profitable project.")

# Calculate IRR using numpy_financial
irr = None
try:
    # npf.irr requires the cash flow series including Year 0
    # Ensure the cash flow series has at least one negative and one positive value to calculate IRR
    has_positive_irr = any(cf > 0 for cf in irr_cash_flows)
    has_negative_irr = any(cf < 0 for cf in irr_cash_flows)

    if has_positive_irr and has_negative_irr:
         # Use a try-except block as npf.irr can raise errors for unusual cash flows
         try:
             irr = npf.irr(irr_cash_flows)
         except Exception as e:
              st.write(f"Internal Rate of Return (IRR) calculation failed: {e}. This can happen with unusual cash flow patterns.")
              irr = None # Ensure irr is None if calculation fails

    else:
         st.write("Internal Rate of Return (IRR): Cannot be calculated (Cash flow series has no sign change)")

except Exception as e:
    # This outer except catches potential errors before calling npf.irr, though less likely now
    st.write(f"An unexpected error occurred during IRR check: {e}")
    irr = None


if irr is not None:
    # Format IRR as a percentage
    st.write(f"Internal Rate of Return (IRR): {irr*100:.2f}%")
else:
     # Cannot be calculated case is handled in the try-except block and checks
     pass


# Visualization
st.subheader("Cash Flow Chart 📊")
st.markdown("Visualize the annual and cumulative cash flows over the project period with interactive controls.")

if irr_cash_flows:
    # Create a DataFrame for easy plotting
    chart_data = pd.DataFrame({
        'Year': list(range(0, int(project_period_years) + 1)),
        'Annual Cash Flow (£)': irr_cash_flows, # Use explicit column names for Plotly
        'Cumulative Cash Flow (£)': np.cumsum(irr_cash_flows)
    })

    # Create figure with secondary y-axis using Plotly Graph Objects
    fig = go.Figure()

    # Add Annual Cash Flow bars
    fig.add_trace(
        go.Bar(
            x=chart_data['Year'],
            y=chart_data['Annual Cash Flow (£)'],
            name='Annual Cash Flow',
            marker_color='skyblue'
        )
    )

    # Add Cumulative Cash Flow line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=chart_data['Year'],
            y=chart_data['Cumulative Cash Flow (£)'],
            mode='lines+markers',
            name='Cumulative Cash Flow',
            yaxis='y2', # Assign to secondary y-axis
            line=dict(color='orange', width=2),
            marker=dict(color='orange', size=8)
        )
    )

    # Update layout for secondary y-axis and overall appearance
    fig.update_layout(
        title_text='Annual vs. Cumulative Cash Flow Over Project Period',
        xaxis_title='Year',
        yaxis_title='Annual Cash Flow (£)',
        yaxis2=dict(
            title='Cumulative Cash Flow (£)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified', # Optional: unified hover for better experience
        legend=dict(x=0.01, y=0.99), # Position legend
        margin=dict(l=0, r=0, t=30, b=0) # Adjust margins if needed
    )

    # Ensure x-axis ticks show integer years
    fig.update_xaxes(tick0=0, dtick=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please set up cash flow data to generate the chart.")