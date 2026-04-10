#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:18:57 2024

@author: gabriel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%% 4.9 GHz resonator

data_folder = Path("Files/corrected data gabriel 04_26/amp dependence")

file_path = data_folder / 'field_dep_4_9_GHz.dat'


# Read the file line by line and manually parse
with open(file_path, 'r') as file:
    lines = file.readlines()

# Define the expected column names
column_names = [
    'Delta_fr_0°', 'Delta_fr_err_0°', 'Delta_n_s_0°', 'Delta_n_s_err_0°',
    'Delta_fr_45°', 'Delta_fr_err_45°', 'Delta_n_s_45°', 'Delta_n_s_err_45°',
    'Delta_fr_90°', 'Delta_fr_err_90°', 'Delta_n_s_90°', 'Delta_n_s_err_90°',
    'Delta_fr_135°', 'Delta_fr_err_135°', 'Delta_n_s_135°', 'Delta_n_s_err_135°',
    'fields_0°', 'fields_45°', 'fields_90°', 'fields_135°'
]

# Parse data rows
data_rows = []
for i, line in enumerate(lines[1:], start=2):  # skip header line
    if not line.strip():
        continue
    
    # Split by whitespace (tabs or spaces)
    values = line.strip().split()
    
    # Convert to float where possible
    row = []
    for v in values:
        try:
            row.append(float(v))
        except ValueError:
            row.append(np.nan)  # Use NaN for empty or invalid values
    
    data_rows.append(row)

# Find the maximum number of columns in any row
max_cols = max(len(row) for row in data_rows)
print(f"Maximum columns found: {max_cols}")
print(f"Expected columns: {len(column_names)}")

# Pad rows that have fewer columns
padded_rows = []
for row in data_rows:
    if len(row) < max_cols:
        row.extend([np.nan] * (max_cols - len(row)))
    padded_rows.append(row)

# Create DataFrame
df = pd.DataFrame(padded_rows)

# The data seems to have extra columns in later rows (likely because some angle measurements stop)
# Let's examine the structure
print(f"DataFrame shape: {df.shape}")

# Based on the file structure, it appears that after a certain point, 
# only the 45° and 135° measurements continue
# Let's create a more accurate parsing approach

# Alternative: Parse by known column positions
data_rows_structured = []
for line in lines[1:]:
    if not line.strip():
        continue
    
    # Split the line
    parts = line.strip().split()
    
    # Initialize row with NaNs
    row = [np.nan] * 20  # 20 expected columns
    
    # Map values based on position (this is approximate and may need adjustment)
    # The first 16 columns are the Delta measurements (4 per angle)
    # The last 4 columns are the field values
    for idx, val in enumerate(parts):
        if idx < 16 and idx < len(parts):
            try:
                row[idx] = float(val)
            except:
                row[idx] = np.nan
        elif idx >= 16:
            # These are the field values
            field_idx = 16 + (idx - 16)
            if field_idx < 20:
                try:
                    row[field_idx] = float(val)
                except:
                    row[field_idx] = np.nan
    
    data_rows_structured.append(row)

# Create DataFrame with proper column names
df_final = pd.DataFrame(data_rows_structured, columns=column_names)

# Display results
print("\nFirst 10 rows:")
print(df_final.head(10))
print(f"\nDataFrame shape: {df_final.shape}")
print(f"\nMissing values per column:")
print(df_final.isnull().sum())

# Save to CSV if needed
# df_final.to_csv('output.csv', index=False)

# View the last few rows to see the structure
print("\nLast 10 rows:")
print(df_final.tail(10))

# df_final.to_excel(data_folder / 'field_dep_4_9_GHz.xlsx')

#%% Plot 4.9 GHz resonator
df_final = pd.read_excel(data_folder / 'field_dep_4_9_GHz.xlsx')
fig, ax = plt.subplots()

ax.errorbar(df_final["fields_0°"], df_final["Delta_fr_0°"], yerr=df_final['Delta_fr_err_0°'], fmt='o')
ax.errorbar(df_final["fields_90°"], df_final["Delta_fr_90°"], yerr=df_final['Delta_fr_err_90°'], fmt='o')
ax.errorbar(df_final["fields_45°"], df_final["Delta_fr_45°"], yerr=df_final['Delta_fr_err_45°'], fmt='o')
ax.errorbar(df_final["fields_135°"], df_final["Delta_fr_135°"], yerr=df_final['Delta_fr_err_135°'], fmt='o')

ax.set_xlabel("B [T]")
ax.set_ylabel(r"$\Delta f$")

#%% 5.7 GHz resonator

data_folder = Path("Files/corrected data gabriel 04_26/amp dependence")

file_path = data_folder / 'field_dep_5_7_GHz.dat'


# Read the file line by line and manually parse
with open(file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Define the expected column names
column_names = [
    'Delta_fr_0°', 'Delta_fr_err_0°', 'Delta_n_s_0°', 'Delta_n_s_err_0°',
    'Delta_fr_45°', 'Delta_fr_err_45°', 'Delta_n_s_45°', 'Delta_n_s_err_45°',
    'Delta_fr_90°', 'Delta_fr_err_90°', 'Delta_n_s_90°', 'Delta_n_s_err_90°',
    'Delta_fr_135°', 'Delta_fr_err_135°', 'Delta_n_s_135°', 'Delta_n_s_err_135°',
    'fields_0°', 'fields_45°', 'fields_90°', 'fields_135°'
]

# Parse data rows
data_rows = []
for i, line in enumerate(lines[1:], start=2):  # skip header line
    if not line.strip():
        continue
    
    # Split by whitespace (tabs or spaces)
    values = line.strip().split()
    
    # Convert to float where possible
    row = []
    for v in values:
        try:
            row.append(float(v))
        except ValueError:
            row.append(np.nan)  # Use NaN for empty or invalid values
    
    data_rows.append(row)

# Find the maximum number of columns in any row
max_cols = max(len(row) for row in data_rows)
print(f"Maximum columns found: {max_cols}")
print(f"Expected columns: {len(column_names)}")

# Pad rows that have fewer columns
padded_rows = []
for row in data_rows:
    if len(row) < max_cols:
        row.extend([np.nan] * (max_cols - len(row)))
    padded_rows.append(row)

# Create DataFrame
df = pd.DataFrame(padded_rows)

# The data seems to have extra columns in later rows (likely because some angle measurements stop)
# Let's examine the structure
print(f"DataFrame shape: {df.shape}")

# Based on the file structure, it appears that after a certain point, 
# only the 45° and 135° measurements continue
# Let's create a more accurate parsing approach

# Alternative: Parse by known column positions
data_rows_structured = []
for line in lines[1:]:
    if not line.strip():
        continue
    
    # Split the line
    parts = line.strip().split()
    
    # Initialize row with NaNs
    row = [np.nan] * 20  # 20 expected columns
    
    # Map values based on position (this is approximate and may need adjustment)
    # The first 16 columns are the Delta measurements (4 per angle)
    # The last 4 columns are the field values
    for idx, val in enumerate(parts):
        if idx < 16 and idx < len(parts):
            try:
                row[idx] = float(val)
            except:
                row[idx] = np.nan
        elif idx >= 16:
            # These are the field values
            field_idx = 16 + (idx - 16)
            if field_idx < 20:
                try:
                    row[field_idx] = float(val)
                except:
                    row[field_idx] = np.nan
    
    data_rows_structured.append(row)

# Create DataFrame with proper column names
df_final = pd.DataFrame(data_rows_structured, columns=column_names)

# Display results
print("\nFirst 10 rows:")
print(df_final.head(10))
print(f"\nDataFrame shape: {df_final.shape}")
print(f"\nMissing values per column:")
print(df_final.isnull().sum())

# Save to CSV if needed
# df_final.to_csv('output.csv', index=False)

# View the last few rows to see the structure
print("\nLast 10 rows:")
print(df_final.tail(10))

# df_final.to_excel(data_folder / 'field_dep_5_7_GHz.xlsx')

#%% Plot 5.7 GHz resonator
df_final = pd.read_excel(data_folder / 'field_dep_5_7_GHz.xlsx')
fig, ax = plt.subplots()

ax.errorbar(df_final["fields_0°"], df_final["Delta_fr_0°"], yerr=df_final['Delta_fr_err_0°'], fmt='o')
ax.errorbar(df_final["fields_90°"], df_final["Delta_fr_90°"], yerr=df_final['Delta_fr_err_90°'], fmt='o')
ax.errorbar(df_final["fields_45°"], df_final["Delta_fr_45°"], yerr=df_final['Delta_fr_err_45°'], fmt='o')
ax.errorbar(df_final["fields_135°"], df_final["Delta_fr_135°"], yerr=df_final['Delta_fr_err_135°'], fmt='o')

ax.set_xlabel("B [T]")
ax.set_ylabel(r"$\Delta f$")
#%% 5.7 GHz resonator

data_folder = Path("Files/corrected data gabriel 04_26/amp dependence")

file_path = data_folder / 'field_dep_9_7_GHz.dat'


# Read the file line by line and manually parse
with open(file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Define the expected column names
column_names = [
    'Delta_fr_0°', 'Delta_fr_err_0°', 'Delta_n_s_0°', 'Delta_n_s_err_0°',
    'Delta_fr_45°', 'Delta_fr_err_45°', 'Delta_n_s_45°', 'Delta_n_s_err_45°',
    'Delta_fr_90°', 'Delta_fr_err_90°', 'Delta_n_s_90°', 'Delta_n_s_err_90°',
    'Delta_fr_135°', 'Delta_fr_err_135°', 'Delta_n_s_135°', 'Delta_n_s_err_135°',
    'fields_0°', 'fields_45°', 'fields_90°', 'fields_135°'
]

# Parse data rows
data_rows = []
for i, line in enumerate(lines[1:], start=2):  # skip header line
    if not line.strip():
        continue
    
    # Split by whitespace (tabs or spaces)
    values = line.strip().split()
    
    # Convert to float where possible
    row = []
    for v in values:
        try:
            row.append(float(v))
        except ValueError:
            row.append(np.nan)  # Use NaN for empty or invalid values
    
    data_rows.append(row)

# Find the maximum number of columns in any row
max_cols = max(len(row) for row in data_rows)
print(f"Maximum columns found: {max_cols}")
print(f"Expected columns: {len(column_names)}")

# Pad rows that have fewer columns
padded_rows = []
for row in data_rows:
    if len(row) < max_cols:
        row.extend([np.nan] * (max_cols - len(row)))
    padded_rows.append(row)

# Create DataFrame
df = pd.DataFrame(padded_rows)

# The data seems to have extra columns in later rows (likely because some angle measurements stop)
# Let's examine the structure
print(f"DataFrame shape: {df.shape}")

# Based on the file structure, it appears that after a certain point, 
# only the 45° and 135° measurements continue
# Let's create a more accurate parsing approach

# Alternative: Parse by known column positions
data_rows_structured = []
for line in lines[1:]:
    if not line.strip():
        continue
    
    # Split the line
    parts = line.strip().split()
    
    # Initialize row with NaNs
    row = [np.nan] * 20  # 20 expected columns
    
    # Map values based on position (this is approximate and may need adjustment)
    # The first 16 columns are the Delta measurements (4 per angle)
    # The last 4 columns are the field values
    for idx, val in enumerate(parts):
        if idx < 16 and idx < len(parts):
            try:
                row[idx] = float(val)
            except:
                row[idx] = np.nan
        elif idx >= 16:
            # These are the field values
            field_idx = 16 + (idx - 16)
            if field_idx < 20:
                try:
                    row[field_idx] = float(val)
                except:
                    row[field_idx] = np.nan
    
    data_rows_structured.append(row)

# Create DataFrame with proper column names
df_final = pd.DataFrame(data_rows_structured, columns=column_names)

# Display results
print("\nFirst 10 rows:")
print(df_final.head(10))
print(f"\nDataFrame shape: {df_final.shape}")
print(f"\nMissing values per column:")
print(df_final.isnull().sum())

# Save to CSV if needed
# df_final.to_csv('output.csv', index=False)

# View the last few rows to see the structure
print("\nLast 10 rows:")
print(df_final.tail(10))

# df_final.to_excel(data_folder / 'field_dep_9_7_GHz.xlsx')

#%% Plot 9.7 GHz resonator
df_final = pd.read_excel(data_folder / 'field_dep_9_7_GHz.xlsx')
fig, ax = plt.subplots()

ax.errorbar(df_final["fields_0°"], df_final["Delta_fr_0°"], yerr=df_final['Delta_fr_err_0°'], fmt='o')
ax.errorbar(df_final["fields_90°"], df_final["Delta_fr_90°"], yerr=df_final['Delta_fr_err_90°'], fmt='o')
ax.errorbar(df_final["fields_45°"], df_final["Delta_fr_45°"], yerr=df_final['Delta_fr_err_45°'], fmt='o')
ax.errorbar(df_final["fields_135°"], df_final["Delta_fr_135°"], yerr=df_final['Delta_fr_err_135°'], fmt='o')

ax.set_xlabel("B [T]")
ax.set_ylabel(r"$\Delta f$")

#%% Temperature dependence 5.7 GHz

data_folder = Path("Files/corrected data gabriel 04_26/temp depend")

file_path = data_folder / 'T_dep_5_7_GHz.dat'


# Read the file line by line and manually parse
with open(file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Define the expected column names
column_names = [
                "T 0mT", "Delta fr 0mT",	 "Delta fr err 0mT", "T 30mT 0°", "Delta fr 30mT 0°", "Delta fr err 30mT 0°",
                "T 30mT 90°", "	Delta fr 30mT 90°", "Delta fr err 30mT 90°", "T 60mT 0°", "Delta fr60mT 0°",
                "Delta fr err60mT 0°", "T 60mT 90°",	"Delta fr 60mT90°", "Delta fr err 60mT 90°"   
                ]

# Parse data rows
data_rows = []
for i, line in enumerate(lines[1:], start=2):  # skip header line
    if not line.strip():
        continue
    
    # Split by whitespace (tabs or spaces)
    values = line.strip().split()
    
    # Convert to float where possible
    row = []
    for v in values:
        try:
            row.append(float(v))
        except ValueError:
            row.append(np.nan)  # Use NaN for empty or invalid values
    
    data_rows.append(row)

# Find the maximum number of columns in any row
max_cols = max(len(row) for row in data_rows)
print(f"Maximum columns found: {max_cols}")
print(f"Expected columns: {len(column_names)}")

# Pad rows that have fewer columns
padded_rows = []
for row in data_rows:
    if len(row) < max_cols:
        row.extend([np.nan] * (max_cols - len(row)))
    padded_rows.append(row)

# Create DataFrame
df = pd.DataFrame(padded_rows)

# The data seems to have extra columns in later rows (likely because some angle measurements stop)
# Let's examine the structure
print(f"DataFrame shape: {df.shape}")

# Based on the file structure, it appears that after a certain point, 
# only the 45° and 135° measurements continue
# Let's create a more accurate parsing approach

# Alternative: Parse by known column positions
data_rows_structured = []
for line in lines[1:]:
    if not line.strip():
        continue
    
    # Split the line
    parts = line.strip().split()
    
    # Initialize row with NaNs
    row = [np.nan] * 15  # 20 expected columns
    
    # Map values based on position (this is approximate and may need adjustment)
    # The first 16 columns are the Delta measurements (4 per angle)
    # The last 4 columns are the field values
    for idx, val in enumerate(parts):
        if idx < 16 and idx < len(parts):
            try:
                row[idx] = float(val)
            except:
                row[idx] = np.nan
        elif idx >= 16:
            # These are the field values
            field_idx = 16 + (idx - 16)
            if field_idx < 20:
                try:
                    row[field_idx] = float(val)
                except:
                    row[field_idx] = np.nan
    
    data_rows_structured.append(row)

# Create DataFrame with proper column names
df_final = pd.DataFrame(data_rows_structured, columns=column_names)

# Display results
print("\nFirst 10 rows:")
print(df_final.head(10))
print(f"\nDataFrame shape: {df_final.shape}")
print(f"\nMissing values per column:")
print(df_final.isnull().sum())

# Save to CSV if needed
# df_final.to_csv('output.csv', index=False)

# View the last few rows to see the structure
print("\nLast 10 rows:")
print(df_final.tail(10))

# df_final.to_excel(data_folder / 'T_dep_5_7_GHz.xlsx')

#%% Plot 5.7 GHz temperature dependence
data_folder = Path("Files/corrected data gabriel 04_26/temp depend")

df_final = pd.read_excel(data_folder / 'T_dep_5_7_GHz.xlsx')
fig, ax = plt.subplots()

ax.errorbar(df_final["T 0mT"], df_final["Delta fr 0mT"], yerr=df_final['Delta fr err 0mT'], fmt='sk')
ax.errorbar(df_final["T 30mT 0°"], df_final["Delta fr 30mT 0°"], yerr=df_final['Delta fr err 30mT 0°'], fmt='sr')
ax.errorbar(df_final["T 30mT 90°"], df_final["Delta fr 30mT 90°"], yerr=df_final['Delta fr err 30mT 90°'], fmt='sr', markerfacecolor="None")
ax.errorbar(df_final["T 60mT 0°"], df_final["Delta fr 60mT 0°"], yerr=df_final['Delta fr err 60mT 0°'], fmt='sb')
ax.errorbar(df_final["T 60mT 90°"], df_final["Delta fr 60mT 90°"], yerr=df_final['Delta fr err 60mT 90°'], fmt='sb', markerfacecolor="None")

ax.set_xlabel(r"$T [mK]$")
ax.set_ylabel(r"$\Delta f$")
