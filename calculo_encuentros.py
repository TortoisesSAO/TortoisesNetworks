import pandas as pd 
import numpy as np
from scipy.spatial.distance import cdist
import pyproj
# estoy queriendo ver que cambia los argumentos extra de linea 30

#if tortoises were close in time and space they go into this consideration 
def save_spacetime_encounters(dfs,max_dist_space=20,max_dist_time=20,file_out="encuentros_Igoto_20min.csv",path="",only_return_df= False): 
    #max dist in space is in meters and the max dist in time is in minutes 
    colum_names=["dateTime_one","dateTime_two","time distance","space distance" ,"name one", "name two","sex one","sex two","lat one","lon one", "lat two", "lon two"]
    df_out=pd.DataFrame(columns=colum_names,dtype=str)
    for i in range(len(dfs)):
        for j in range(i+1,len(dfs)):
            if dfs[i]["t_name"].iloc[0]!=dfs[j]["t_name"].iloc[0]:
                df_out=find_near_spacetime_points(dfs[i],dfs[j],max_dist_time,max_dist_space,df_out)
    if only_return_df:
        return df_out
    else:
        df_out.to_csv(path+file_out,index=False,sep=";")

#finds all points that are close in time and space on a day
def find_near_spacetime_points(df1,df2,max_dist_time,max_dist_space,df_out):# max_dis
    points1=get_cordinates(df1)
    points2=get_cordinates(df2)
    # create projections, using a mean (lat, lon) for aeqd
    lat_0, lon_0 = np.mean(np.append(points1[:,0], points2[:,0])), np.mean(np.append(points1[:,1], points2[:,1]))
    proj = pyproj.Proj(proj='aeqd', lat_0=lat_0, lon_0=lon_0, x_0=lon_0, y_0=lat_0,units="m")
    WGS84 = pyproj.Proj(init='epsg:4326')
    # transform coordinates
    projected_points1 = pyproj.transform(WGS84, proj, points1[:,1], points1[:,0])
    projected_points2 = pyproj.transform(WGS84, proj, points2[:,1], points2[:,0])
    projected_points1 = np.column_stack(projected_points1)
    projected_points2 = np.column_stack(projected_points2)
    distances = cdist(projected_points1, projected_points2)
    # Find the pairs of points that are close in time and space
    indexes = np.where((distances < max_dist_space) & (np.abs(df1['dateTime'].values[:, None] - df2['dateTime'].values) / np.timedelta64(1, 'm') < max_dist_time))
    points1_indexes, points2_indexes = indexes
    # Select the rows of df1 and df2 that correspond to the points in points1 and points2, respectively
    df1_selected = df1.iloc[points1_indexes]
    df2_selected = df2.iloc[points2_indexes]
    # Compute the time differences between the selected points in df1 and df2
    dt = np.abs(df1_selected['dateTime'].values - df2_selected['dateTime'].values) / np.timedelta64(1, 'm')
    # Create a dictionary with the results
    results = {
        "dateTime_one": df1_selected['dateTime'].values,
        "dateTime_two": df2_selected['dateTime'].values,
        "time distance": dt.astype(str),
        "space distance": distances[indexes].astype(str),
        "name one": df1['t_name'].iloc[0],
        "name two": df2['t_name'].iloc[0],
        "sex one": df1['sex'].iloc[0],
        "sex two": df2['sex'].iloc[0],
        "lat one": points1[points1_indexes, 0],
        "lon one": points1[points1_indexes, 1],
        "lat two": points2[points2_indexes, 0],
        "lon two": points2[points2_indexes, 1],
    }
    # Append the results to df_out
    df_out = pd.concat([df_out, pd.DataFrame(results)], ignore_index=True)
    return df_out

def get_cordinates(df):
    points=zip(df["lat"],df["lon"])
    points=np.array(list(points))
    return points

#gets difference in time from points from a day in dataframes
def get_delta_time(df1,df2,point1,point2):
    t1=df1.loc[(df1["lat"]==point1[0]) & (df1["lon"]==point1[1])]["dateTime"]
    t2=df2.loc[(df2["lat"]==point2[0]) & (df2["lon"]==point2[1])]["dateTime"]
    t1=t1.to_numpy()
    t2=t2.to_numpy()
    dt=((t1[0]-t2[0]).astype('timedelta64[s]')).astype("int")
    return np.abs(dt/60),t1[0],t2[0]



""" dfs_igos = leer_archivos_data.get_igo_old_files()
max_dist_space_ = 20
max_dist_time_ = 20
df_encounters_igos = save_spacetime_encounters(dfs_igos,max_dist_space=max_dist_space_,max_dist_time=max_dist_time_,only_return_df=True)
ipdb.set_trace() """