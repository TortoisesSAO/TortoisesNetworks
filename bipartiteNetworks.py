import pandas as pd 
import glob
import numpy as np
import folium
import folium.plugins
import ipdb
from geopy import distance 
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import os
import mantel
import random
from html2image import Html2Image
from PIL import Image
import math

#saves all refugies information into a dataframe
def save_refugies_data(dfs,start_time="20:00",end_time="06:00" ,distance_refugies=20):
    df_out=pd.DataFrame(columns=["lat","lon","date","t_name","sex","refugie_label","mean_distance_points","num_points","max_distance_from_refugie"])
    refugies=[]
    start_time = datetime.datetime.strptime(start_time, "%H:%M").time()
    end_time = datetime.datetime.strptime(end_time, "%H:%M").time()
    for j in range(len(dfs)):
        dates = dfs[j]['dateTime'].dt.date
        dates = dates.unique()
        for date in dates:
            start_datetime = datetime.datetime.combine(date, start_time)
            end_datetime = datetime.datetime.combine(date + datetime.timedelta(days=1), end_time)
            df_aux=dfs[j][(dfs[j]["dateTime"]>=start_datetime) & (dfs[j]["dateTime"]<=end_datetime)]
            if len(df_aux)>=1:
                lat_r,lon_r = df_aux["lat"].mean(),df_aux["lon"].mean()
                #check the distance between the refugie and the other refugies already found
                in_refugies,refugie=poin_in_refuguies((lat_r,lon_r),refugies,distance_refugies)
                if in_refugies:
                    # make mean distance to refugie with data in df_aux
                    dist_point_ref=df_aux.apply(lambda x: distance.distance((x["lat"],x["lon"]),refugie).meters,axis=1)
                    mean_dist_to_ref=dist_point_ref.mean()
                    max_dist_to_ref=dist_point_ref.max()
                    new_row = {
                    "lat": refugie[0],
                    "lon": refugie[1],
                    "date": date,
                    "t_name": dfs[j]["t_name"].iloc[0],
                    "sex": dfs[j]["sex"].iloc[0],
                    "refugie_label": refugies.index(refugie),
                    "mean_distance_points": mean_dist_to_ref,
                    "num_points": len(df_aux),
                    "max_distance_from_refugie": max_dist_to_ref
                    }
                else:
                    refugies.append((lat_r,lon_r))
                    dist_point_ref=df_aux.apply(lambda x: distance.distance((x["lat"],x["lon"]),(lat_r,lon_r)).meters,axis=1)
                    mean_dist_to_ref=dist_point_ref.mean()
                    max_dist_to_ref=dist_point_ref.max()
                    new_row = {
                    "lat": lat_r,
                    "lon": lon_r,
                    "date": date,
                    "t_name": dfs[j]["t_name"].iloc[0],
                    "sex": dfs[j]["sex"].iloc[0],
                    "refugie_label": refugies.index((lat_r,lon_r)),
                    "mean_distance_points": mean_dist_to_ref,
                    "num_points": len(df_aux),
                    "max_distance_from_refugie": max_dist_to_ref
                    }
                new_row = pd.DataFrame(new_row,index=[0])
                df_out = pd.concat((df_out,new_row), ignore_index=True)
    return df_out

#check if the point is considered a new refugie or not        
def poin_in_refuguies(points,refugies,distance_refugies):
    for refuguie in refugies:
        if distance.distance(points,refuguie).meters<=distance_refugies:
            return True,refuguie
    return False,0


#makes html map with the refugies   
def make_map_from_refuguies(df_ref,topo_map=False,radius_nodes=10,refus_labels=False,refus_labels_anchor = (0,0)):
    map_out=get_map(topo_map)
    # get refugies_label as sorted as it is in refugies
    refugies_label=df_ref["refugie_label"].unique()
    refugies_label.sort() 
    for i in range(len(refugies_label)):
        refugie=df_ref[df_ref["refugie_label"]==refugies_label[i]].iloc[0][["lat","lon"]]
        df_aux=df_ref[df_ref["refugie_label"]==refugies_label[i]]
        t_names=np.unique(df_aux["t_name"].values)
        folium.CircleMarker(location=[refugie[0],refugie[1]],radius=radius_nodes,color="orange",fill_color="orange",fill_opacity=0.3,popup="<b>Refugio</b><br>"+"nro"+str(refugies_label[i])+"  "+str(t_names)).add_to(map_out)
        # add not popup text to the map 
        if refus_labels:
            folium.Marker(location=[refugie[0],refugie[1]],icon=folium.features.DivIcon(icon_size=(150,36),icon_anchor=refus_labels_anchor,html='<div style="font-size: 10pt; color: white">%s</div>' % str(refugies_label[i]))).add_to(map_out)
    return map_out



def get_map(topo_map=False,coords=[-40.585390,-64.996220],zoom=15):
    map1 = folium.Map(location = coords,zoom_start=zoom)
    if topo_map:
        folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attr= 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)').add_to(map1)
    else: 
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community").add_to(map1)
    return map1 



# makes bipartite network from nodes ref and nodes turltes
def get_bigraph(df_ref,plot=False,k=0.5,return_refugies=False,nodesize=200,scale=1,iters=50,weight="weight",log_weights=False,log_base=20,period_filter=False,month_start="01",month_end="12"):
    #if period_filter:
    # filter data from start_month to end month, if month end is less than month start, give the vuelta 
    refugies_label=df_ref["refugie_label"].unique()
    refugies = [df_ref[df_ref["refugie_label"]==refugies_label[i]].iloc[0][["lat","lon"]] for i in range(len(refugies_label))]
    t_uniq_names=np.unique(df_ref["t_name"].values)
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    refuguies_nodes=refugies_label
    
    B.add_nodes_from(t_uniq_names.tolist(), bipartite=1)
    B.add_nodes_from(refuguies_nodes.tolist(), bipartite=0)
    # Add edges with the edge attribute "weight"
    for i in range(len(refugies)):
        refugie=refugies[i]
        df_aux=df_ref[(df_ref["lat"]==refugie[0]) & (df_ref["lon"]==refugie[1])]
        for t_name in t_uniq_names:  
            if len(df_aux[df_aux["t_name"]==t_name])>0: 
                B.add_edge(t_name,refuguies_nodes[i],weight=len(df_aux[df_aux["t_name"]==t_name]))
    if plot:
        edges = B.edges()
        weights = [B[u][v]['weight'] for u,v in edges]
        weights=np.array(weights)
        weights=5*weights/np.max(weights)+np.ones(len(weights))*0.1
        colors_refugies=["sandybrown"]*len(refugies)
        colors_t_names=get_colors_turtles(df_ref,t_uniq_names)
        if log_weights:
            #change weights to log scale in B 
            for edge in B.edges():
                B.edges[edge]["weight"]=np.log(B.edges[edge]["weight"]+log_base)/np.log(log_base)
        pos=nx.spring_layout(B,k,iterations=iters,scale=scale,weight=weight)
        nx.draw_networkx_edges(B, pos=pos, width=weights)
        nx.draw_networkx_nodes(B, pos=pos, nodelist=refuguies_nodes, node_color=colors_refugies,node_size=nodesize,label=refuguies_nodes)
        nx.draw_networkx_nodes(B, pos=pos, nodelist=t_uniq_names, node_color=colors_t_names,node_size=nodesize,label=t_uniq_names)
        nx.draw_networkx_labels(B,pos,font_size=10,font_family='sans-serif')
        plt.show()
    if return_refugies:
        return B,refugies
    return B


def get_colors_turtles(df_ref,t_uniq_names):
    t_colors=[]
    for turtle in t_uniq_names:
        sex=df_ref[df_ref["t_name"]==turtle]["sex"].iloc[0]
        if sex=="macho":
            t_colors.append("lightblue")
        elif sex=="hembra":
            t_colors.append("pink")
        else:
            t_colors.append("silver")
    return t_colors

""" folder_windows_N = "C:/Users/marco/facultad/tesis/MaestriaMarco/DataAnalysis/Datos_CampanaSaoFeb/Ns"
pickle_dict_windows = "C:/Users/marco/facultad/tesis/MaestriaMarco/sex_dict_tortoises.pickle"
dfs_N = leer_archivos_data.get_N_files(folder_windows_N, pickle_dict_windows,filter=False)
dfs_N_filtered = leer_archivos_data.get_N_files(folder_windows_N, pickle_dict_windows,filter=True)
df_refugios_N = save_refugies_data(dfs_N_filtered)
ipdb.set_trace() """