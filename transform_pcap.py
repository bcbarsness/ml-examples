import pandas as pd
import json
import re

pcap_data = pd.read_csv('packet_metadata.csv', index_col='No.')

dataframe = pcap_data
src_dst = dataframe[["Source","Destination"]]
src_dst.sample(10)

def ip_matcher(address):
    # Used to validate if string is an ipaddress
    ip = re.match(
        '^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$', address)
    if ip:
        return True
    else:
        return False

src_dst.rename(columns={"Source":"source","Destination":"target"}, inplace=True)
src_dst['valid_src'] = src_dst.source.apply(ip_matcher)
src_dst['valid_target'] = src_dst.target.apply(ip_matcher)

valid_src_dest = src_dst[(src_dst.valid_src==True) & (src_dst.valid_target==True)]

grouped_src_dst = valid_src_dest.groupby(["source","target"]).size().reset_index()

unique_ips = pd.Index(grouped_src_dst['source']
                      .append(grouped_src_dst['target'])
                      .reset_index(drop=True).unique())

group_dict = {}
counter = 0
for ip in unique_ips:
    breakout_ip = re.match("^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$", ip)
    if breakout_ip:
        net_id = '.'.join(breakout_ip.group(1,2,3))
        if net_id not in group_dict:
            counter += 1
            group_dict[net_id] = counter
        else:
            pass

grouped_src_dst.rename(columns={0:'count'}, inplace=True)
temp_links_list = list(grouped_src_dst.apply(lambda row: {"source": row['source'], "target": row['target'], "value": row['count']}, axis=1))

links_list = []
for link in temp_links_list:
    record = {"value":link['value'], "source":unique_ips.get_loc(link['source']),
     "target": unique_ips.get_loc(link['target'])}
    links_list.append(record)

nodes_list = []
for ip in unique_ips:
    breakout_ip = re.match("^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$", ip)
    if breakout_ip:
        net_id = '.'.join(breakout_ip.group(1,2,3))
        nodes_list.append({"name":ip, "group": group_dict.get(net_id)})

json_prep = {"nodes":nodes_list, "links":links_list}

#json_prep.keys().dict_keys(['links', 'nodes'])

json_dump = json.dumps(json_prep, indent=1, sort_keys=True)

filename_out = 'pcap_export.json'
json_out = open(filename_out,'w')
json_out.write(json_dump)
json_out.close()

#print(json_dump)

#print(nodes_list[0:8])
#print (links_list)
#print(temp_links_list)
#print(dataframe.head())
#print(src_dst)
