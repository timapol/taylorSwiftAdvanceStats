
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
from os.path import exists
from pathlib import Path
import mplcursors
import numpy as np

p = Path(__file__).parent.absolute()


CLIENT_ID = '2d3331597dee44dcb89c61baffb42ff2'
CLIENT_SECRET = 'f1479d8c41944a19a62eadfab2eb0404'

AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'

#user profile
#https://open.spotify.com/user/123081269?si=f1a48bbc25bb42e6
fields_block1 = {
    #'fields':'tracks.items(track(name,href,album(name,href)))'
    #'fields':'tracks.items(track(name,album(name)))',
    'offset':'0'
}

fields_block2 = {
    #'fields':'tracks.items(track(name,href,album(name,href)))'
    #'fields':'tracks.items(track(name,album(name)))',
    'offset':'100'
}

albumPlotMap = {'Taylor Swift':['*','#c09a79'], 'Speak Now':['D','#904275'], 'Fearless':['.','#886633'],  'Red':['x','#962c2e'], '1989':['P','#4b4952'], 'reputation':['h','b'], 'Lover':['v','#ffcee0'], 'folklore':['8','#5c5c5c'], 'evermore':['p','#68AC41'], 'Midnights':['1','#C28068']}

playlistId_tim      = '0LgZ0fq02lPnape3Y2lBIL' #source ranking
playlist_id_SANDBOX = '1bfrHHGVc5gbJyLmfnK7t2' #sandbox playlist
playlist_inOrder    = '6wrTcAyc3n9tXuiWroLGIu'

class songRanking:
    def __init__(self, title, album, initialRank, uri, popularity, rankSource):
        self.title = title.split("(")[0].strip()
        self.album = album.split("(")[0].strip()
        self.ranking = {rankSource : initialRank}
        self.differenceMap = {}
        self.uri = uri.split(':')[-1]
        self.popularity = popularity
        self.featuresSet = False
        self.numSources = 1

    def __str__(self):
        stringRep = str(self.title) + " "
        for key in self.ranking:
            stringRep = stringRep + " " + key + ":" + str(self.ranking[key])
        return stringRep

    def addFeatures(self):
        if not self.featuresSet:
            r2 = requests.get(BASE_URL + 'audio-features/' + self.uri, headers=headers)

            self.advFeatures = r2.json()
            self.featuresSet = True
            return
        print("Refused to set the same features object twice: ", self.title)
        #input()
    
    def addRanking(self, rank, rankSource):
        if rankSource in self.ranking:
            print("Refused to add the same person's ranking twice: ", rankSource, self.title)
            #input()
            return
        self.ranking[rankSource] = rank
        self.numSources = self.numSources + 1 
        
    def generateDifferenceMap(self):
        #print(self.title)
        if self.numSources < 2:
            print("Cannot generate difference: ", self.ranking, self.title)
            return
        for i,(source1, rank1) in enumerate(self.ranking.items()):
            if i < (len(self.ranking.items()) - 1):
                for source2,rank2 in list(self.ranking.items())[i+1:]:
                    self.differenceMap[source1+source2] = rank1 - rank2

def getPlaylistData(masterRankingsDic, advancedFeaturesFromFile, playlistId, rankSource):
    print("Get from ", rankSource)
    r = requests.get(BASE_URL + 'playlists/' + playlistId + '/tracks', headers=headers,params=fields_block1)
    r = r.json()
    rank = 0
    for trackCont in r['items']:
        album = trackCont['track']['album']['name']
        track = trackCont['track']['name']

        uri = trackCont['track']['uri']
        popularity = trackCont['track']['popularity']
        
        if track in masterRankingsDic:
            masterRankingsDic[track].addRanking(rank, rankSource)
            rank = rank+1
            continue
        ranking = songRanking(track, album, rank, uri, popularity, rankSource)
        rank = rank+1
        if not advancedFeaturesFromFile:
            ranking.addFeatures()
        masterRankingsDic[track] = ranking

    r = requests.get(BASE_URL + 'playlists/' + playlistId + '/tracks', headers=headers,params=fields_block2)
    r = r.json()
    for trackCont in r['items']:
        album = trackCont['track']['album']['name']
        track = trackCont['track']['name']
        uri = trackCont['track']['uri']
        popularity = trackCont['track']['popularity']

        if track in masterRankingsDic:
            masterRankingsDic[track].addRanking(rank, rankSource)
            rank = rank+1
            continue
        
        ranking = songRanking(track, album, rank, uri, popularity, rankSource)
        rank = rank+1
        if not advancedFeaturesFromFile:
            ranking.addFeatures()
        masterRankingsDic[track] = ranking
        
    print("Imported " + rankSource + "\'s rankings, " + str(rank) + " songs") 

def temp_clonePlaylist(masterRankingsDic):
    print("Can't seem to get POST working")
    return
    cloneList = []
    for song, ranking in songRankings.items():
        cloneList.append((ranking.ranking['tim'], ranking.uri))
    cloneList.sort(reverse=True)
    
    

    for song in cloneList:
        uriString = 'spotify:track:' + song[1]
        print(song[1])
        fields_addSong = {
            #'fields':'tracks.items(track(name,href,album(name,href)))'
            #'fields':'tracks.items(track(name,album(name)))',
            'uris':uriString
        }

        r = requests.post(BASE_URL + 'playlists/' + playlist_id_SANDBOX  + '/tracks',headers=headers,params=fields_addSong)
        print(r.content)
        sleep(0.01)

def plot2d(masterRankingsDic):
    print("plot2d depreciated. use pandas idiot")
    return
    ratings1 = []
    ratings2 = []
    
    albumsUsed = set()

    for title,container in masterRankingsDic.items():
        if container.numSources > 1:
            (name1,rank1) , (name2,rank2) = list(container.ranking.items())
            ratings1.append(rank1)
            ratings2.append(rank2)
            if not container.album in albumPlotMap:
                print("No album match-" + container.album + "-" + container.title + " " +str(container.ranking))
                plot(rank1,rank2,'.')
                continue
            if not container.album in albumsUsed:
                albumsUsed.add(container.album)
                plt.plot(rank1,rank2, marker=albumPlotMap[container.album][0], markerfacecolor=albumPlotMap[container.album][1], markeredgecolor=albumPlotMap[container.album][1], lw=0, label=container.album)
                continue
            plt.plot(rank1,rank2, marker=albumPlotMap[container.album][0], markerfacecolor=albumPlotMap[container.album][1], markeredgecolor=albumPlotMap[container.album][1], lw=0)
    
    plt.plot(ratings1,ratings1,'black')

    plt.legend()
    plt.show()

def saveAdvancedFeatures(masterRankingsDic, haveAdvancedFeatures):
    if haveAdvancedFeatures:
        return pd.read_csv("C:\\Users\\timap\\Documents\\pythonMess\\taylor\\advFeatures.csv")
    i = -1
    for title,container in masterRankingsDic.items():
        i = i + 1
        pdRow = {"title":title, 'danceability':container.advFeatures['danceability'],           
                                'energy':container.advFeatures['energy'],                       
                                'key':container.advFeatures['key'],                             
                                'loudness':container.advFeatures['loudness'],                   
                                'mode':container.advFeatures['mode'],                           
                                'speechiness':container.advFeatures['speechiness'],             
                                'acousticness':container.advFeatures['acousticness'],           
                                'instrumentalness':container.advFeatures['instrumentalness'],   
                                'liveness':container.advFeatures['liveness'],                   
                                'valence':container.advFeatures['valence'],                     
                                'tempo':container.advFeatures['tempo'],                         
                                'duration_ms':container.advFeatures['duration_ms'],             
                                'uri':container.advFeatures['uri']}
        if i == 0:
            df = pd.DataFrame(pdRow, index=[0])
            continue
        df.loc[i] = pdRow
    df.to_csv("C:\\Users\\timap\\Documents\\pythonMess\\taylor\\advFeatures.csv")
    return df
    
def repairSins(df_base, songRankings, rankers):
    albumList = []
    df_base["album"] = [songRankings[song].album for song in df_base["title"]]
    df_base["popularity"] = [songRankings[song].popularity for song in df_base["title"]]
    rankExtract = {person:[] for person in rankers}
    for song in df_base["title"]:
        [rankExtract[person].append(songRankings[song].ranking[person] if person in songRankings[song].ranking else len(songRankings)) for person in rankers]
    for person in rankExtract:
        df_base[person] = rankExtract[person]
    for i,person in enumerate(rankExtract):
        if i < (len(rankExtract) - 1):
            for j,(person2,rank2) in enumerate(list(rankExtract.items())):
                if j > i:
                    df_base[person + "_" + person2] = df_base[person] - df_base[person2]
    return df_base

def corr(df,rankers):
    advancedFeatures = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms','popularity']
    for person in rankers:
        print(person + ":")
        [print("\t",item) for item in zip(rankers,[np.corrcoef(df.iloc[:,df.columns.get_loc(person2)], df.iloc[:,df.columns.get_loc(person)])[0,1]**2 for person2 in rankers])]
        print()
        [print("\t",item) for item in zip(advancedFeatures,[np.corrcoef(df.iloc[:,df.columns.get_loc(feature) ], df.iloc[:,df.columns.get_loc(person)])[0,1]**2 for feature in advancedFeatures])]

def accumulatedRanking(df,rankers):
    #df.iloc[:,[df.columns.get_loc(person) for person in rankers]]
    df_summed = df.iloc[:,[df.columns.get_loc(person) for person in rankers]].sum(axis=1)
    df_averaged = df_summed.apply(lambda x : x / len(rankers))
    df['AvRank'] = df_averaged
    df_AvSorted = df.sort_values(by=['AvRank']).iloc[:,[df.columns.get_loc('title'),df.columns.get_loc('AvRank')]]
    print(df_AvSorted.head(20))
    print(df_AvSorted.tail(20))
    
    return df_averaged
    
#handle this? df[df['album'] not in albumPlotMap]
def plotPan(df,rankers):
    for i,person1 in enumerate(rankers):
        if i < (len(rankers) - 1):
            for j,person2 in enumerate(rankers):
                if j > i:
                    #axis = None
                    for album in albumPlotMap:
                        axis = None
                        df_albumFilter = df[df["album"] == album]
                        axis = df_albumFilter.plot(
                            x=person1,
                            y=person2,
                            kind='scatter',
                            color=albumPlotMap[album][1],
                            marker=albumPlotMap[album][0],
                            ax=axis,
                            label=album
                        )
                        #matplotlib.text.Annotation(df_albumFilter.title, xy=(df_albumFilter.tim, df_albumFilter.kay))
                        #mplcursors.cursor(df_albumFilter.title,annotation_kwargs=")
                        cursor = mplcursors.cursor(axis)
                        #cursor.connect("add", lambda sel: sel.annotation.set_text(df_albumFilter.title[sel.index]))
                        #cursor.connect("add", lambda sel: sel.annotation.set_text(sel.target.__dict__))
                        #cursor.connect("add", 
                        #                lambda sel: sel.annotation.set_text(
                        #                df_albumFilter[int( sel.target.data[0])].title)
                        #               )
                        #cursor.connect("add", lambda sel: sel.annotation.set_text(getTitleInLambda(df_albumFilter,0)))
                        cursor.connect("add", lambda sel: sel.annotation.set_text(getTitleInLambda(df_albumFilter,sel.target.data[0])))
                        axis = plt.plot(range(0,len(df.index)),range(0,len(df.index)),color='black')
                        plt.legend()
                        #plt.canvas.mpl_connect('motion_notify_event', on_plot_hover) 
                    plt.show()
    #plt.show()

def getTitleInLambda(df,index):
    #return ("test" + str(index))
    try:
        #print(df)
        #print("index" + str(index))
        #result = df.iloc[int(index)].title
        result = df[df["tim"] == int(index)]
        #print()
        return result["title"].item()
    except IndexError:
        print('index error')
        return "errored"

songRankings = {}

haveAdvancedFeatures = exists(str(p) + "//advFeatures.csv")
inputMap = {"tim":'0LgZ0fq02lPnape3Y2lBIL', 'kay':'3GnsHYtgqTb8wACbhJK5TZ'}
#inputMap = {"tim":'0LgZ0fq02lPnape3Y2lBIL', 'inOrder':'6wrTcAyc3n9tXuiWroLGIu', 'inOrder2':'6wrTcAyc3n9tXuiWroLGIu'}

print("Getting Playlist Data")
[getPlaylistData(songRankings, haveAdvancedFeatures, inputMap[person], person) for person in inputMap]
print("Saving Advanced Features")
df_advFeatures = saveAdvancedFeatures(songRankings, haveAdvancedFeatures)
print("Recombobulating...")
df_all = repairSins(df_advFeatures, songRankings, inputMap.keys())

print("Plotting...")
#print(df_all)
#corr(df_all, inputMap.keys())
#accumulatedRanking(df_all, inputMap.keys())

#print(df_all.iloc[int(98)].title)
plotPan(df_all,inputMap.keys())

#comparison of large differences in rating between people?

