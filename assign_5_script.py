# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Assignment # 5: Netflow Packet Analysis

# %%
# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


# %%
# Read network data into dataframe
network_data = pd.read_csv('Netflow_dataset.csv')

# %% [markdown]
# ## a) Average Packet sizes

# %%
# a. Average size of the packets across all the traffic captured in the dataset
num_packets = network_data['dpkts'].sum()
num_bytes = network_data['doctets'].sum()
avg_packet_size = num_bytes/num_packets

print("Average Packet size:", avg_packet_size)

# %% [markdown]
# ## b) Complementary Cumulative Probability Distribution (CCDF)

# %%
# Calculate flow durations
network_data['duration'] = network_data['last'] - network_data['first']


# %%
# Plot CCDF of durations
plt.hist(network_data['duration'], bins=50, density=True, histtype='step', cumulative=-1)

# Set labels
plt.xlabel('Duration')
plt.ylabel('Probability')
plt.title('Duration CCDF')
plt.show()


# %%
# Plot CCDF of durations with log scale
plt.hist(network_data['duration'], bins=50, density=True, histtype='step', cumulative=-1, log=True)

# Set labels
plt.xlabel('Duration')
plt.ylabel('Probability')
plt.title('Duration CCDF (log scale)')
plt.show()


# %%
# Plot CCDF of durations with log scale
plt.hist(network_data['doctets'], bins=50, density=True, histtype='step', cumulative=-1)

# Set labels
plt.xlabel('# of bytes')
plt.ylabel('Probability')
plt.title('# of bytes CCDF')
plt.show()


# %%
# Plot CCDF of durations with log scale
plt.hist(network_data['doctets'], bins=50, density=True, histtype='step', cumulative=-1, log=True)

# Set labels
plt.xlabel('# of bytes')
plt.ylabel('Probability')
plt.title('# of bytes CCDF (log scale)')
plt.show()


# %%
# Plot CCDF of durations with log scale
plt.hist(network_data['dpkts'], bins=50, density=True, histtype='step', cumulative=-1)

# Set labels
plt.xlabel('# of packets')
plt.ylabel('Probability')
plt.title('# of packets CCDF')
plt.show()


# %%
# Plot CCDF of durations with log scale
plt.hist(network_data['dpkts'], bins=50, density=True, histtype='step', cumulative=-1, log=True)

# Set labels
plt.xlabel('# of packets')
plt.ylabel('Probability')
plt.title('# of packets CCDF (log scale)')
plt.show()

# %% [markdown]
# ## c) Kind of traffic flowing through router

# %%
# We only need these 2 colums
sender_traffic = network_data[['srcport', 'doctets']]

# Create table of port frequencies
src_port_frequency = sender_traffic['srcport'].value_counts().head(10).to_frame(name='frequency')

# Create table of port usage in bytes
src_port_bytes = sender_traffic.groupby(['srcport']).sum()

# Add percentage column to port usage by byte
src_port_bytes['doctets_perc'] = (src_port_bytes['doctets'] / src_port_bytes['doctets'].sum()) * 100

# Merge tables
src_port_frequency.merge(src_port_bytes, left_index=True, right_on='srcport')


# %%
# We only need these 2 colums
receiver_traffic = network_data[['dstport', 'doctets']]

# Create table of port frequencies
dst_port_frequency = receiver_traffic['dstport'].value_counts().head(10).to_frame(name='frequency')

# Create table of port usage in bytes
dst_port_bytes = receiver_traffic.groupby(['dstport']).sum()

# Add percentage column to port usage by byte
dst_port_bytes['doctets_perc'] = (dst_port_bytes['doctets'] / dst_port_bytes['doctets'].sum()) * 100

# Merge tables
dst_port_frequency.merge(dst_port_bytes, left_index=True, right_on='dstport')

# %% [markdown]
# ## d) Traffic volumes based on source IP prefix

# %%
def top_percents(src_addr_volume):
    dec_perc = 0.001
    for dec_perc in [0.001, 0.01, 0.1]:
        percentage = src_addr_volume.iloc[int(len(src_addr_volume) * dec_perc)]['doctets_cumperc']
        print("Top", dec_perc * 100, "\b% of IP addresses:", percentage, "\b% of all bytes")
        dec_perc *= 10


# %%
# Get source IP addresses bytes used by each
src_addr_volume = network_data[['srcaddr', 'doctets']].groupby('srcaddr').sum().sort_values(by='doctets', ascending=False)

# Add percentage of total bytes that each IP address uses
src_addr_volume['doctets_perc'] = (src_addr_volume['doctets'] / src_addr_volume['doctets'].sum()) * 100

# A cummulative percentage column
src_addr_volume['doctets_cumperc'] = src_addr_volume['doctets_perc'].cumsum()

# Get byte percentage by top 0.1%, 1%, and 10% of IP addresses
top_percents(src_addr_volume)


# %%
# Get byte volume by source mask
src_mask_volume = network_data[['src_mask', 'doctets']].groupby('src_mask').sum().sort_index()

# Add percentage of whole bytes for each mask
src_mask_volume['doctets_perc'] = (src_mask_volume['doctets'] / src_mask_volume['doctets'].sum()) * 100

# Get the 0 mask entry
mask_length_zero_perc = src_mask_volume.query('src_mask == 0')['doctets_perc'][0]

# Print the percentage
print("Percentage of traffic with source mask of 0:", mask_length_zero_perc, "\b%")


# %%
# Get source IP addresses bytes used by each, but exclude 0 src_mask
src_addr_volume = network_data[['srcaddr', 'doctets', 'src_mask']].query('src_mask != 0').groupby('srcaddr').sum().sort_values(by='doctets', ascending=False)

# Add percentage of total bytes that each IP address uses
src_addr_volume['doctets_perc'] = (src_addr_volume['doctets'] / src_addr_volume['doctets'].sum()) * 100

# A cummulative percentage column
src_addr_volume['doctets_cumperc'] = src_addr_volume['doctets_perc'].cumsum()

# Get byte percentage by top 0.1%, 1%, and 10% of IP addresses
print("Excluding 0 source masks...")
top_percents(src_addr_volume)

# %% [markdown]
# ## e) Institution with 128.112.0.0/16 address block

# %%
# Grab source addresses, # of packets, and # of bytes
inst_data = network_data[['srcaddr', 'dpkts', 'doctets']]

# True if source address is in address block 128.112
inst_data['from_institution'] = inst_data['srcaddr'].str.startswith('128.112')

# Find # of packets and # of bytes sent by in and out of institution 
inst_data = inst_data.groupby('from_institution').sum()

inst_data['dpkts_perc'] = (inst_data['dpkts'] / inst_data['dpkts'].sum()) * 100

inst_data['doctets_perc'] = (inst_data['doctets'] / inst_data['doctets'].sum()) * 100

print('Traffic sent from institution...')
inst_data


# %%
# Grab source addresses, # of packets, and # of bytes
inst_data = network_data[['dstaddr', 'dpkts', 'doctets']]

# True if source address is in address block 128.112
inst_data['to_institution'] = inst_data['dstaddr'].str.startswith('128.112')

# Find # of packets and # of bytes sent by in and out of institution 
inst_data = inst_data.groupby('to_institution').sum()

inst_data['dpkts_perc'] = (inst_data['dpkts'] / inst_data['dpkts'].sum()) * 100

inst_data['doctets_perc'] = (inst_data['doctets'] / inst_data['doctets'].sum()) * 100

print('Traffic Sent to institution...')
inst_data


