import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import relevant Classes and make matplotlib inline to avoid loading plots at runtime
from math import ceil
from ipyleaflet import GeoJSON
from ipywidgets import Button, VBox, HBox, Box, Layout, Output, Text, Label

import matplotlib.pyplot as plt
import pickle as pkl
import functools
import time
import folium

import ee
ee.Initialize()

import sys
"""
sys.path.append("../Configs/")
import constants
"""

def loadGroup(imageDict,hrImgDict,groups,m,userNameValue,confirmed,denied,edit,add,constants):
    #Create Button Click Events
    def confirmKiln(b,buttons,kilnType):
        for button in buttons:
            if button.style.button_color != 'gray':
                button.style.button_color='gray'
                
        if (b.tooltip in denied.value.keys()):
            edit.value[b.tooltip]={'add':kilnType,'data':{"time":time.time(),"user":userNameValue.value,"kilnType":kilnType}}
        elif b.tooltip in edit.value:
            edit.value[b.tooltip]={'add':kilnType,'data':{"time":time.time(),"user":userNameValue.value,"kilnType":kilnType}}
        elif (b.tooltip not in confirmed.value.keys()):
            add.value[b.tooltip]={'add':kilnType,'data':{"time":time.time(),"user":userNameValue.value,"kilnType":kilnType}}
        b.style.button_color='green'
            
    def denyKiln(b,buttons):
        for button in buttons:
            if button.style.button_color != 'gray':
                button.style.button_color='gray'
                
        if (b.tooltip in confirmed.value.keys()):
            edit.value[b.tooltip]={'add':'Denied','data':{"time":time.time(),"user":userNameValue.value}}
        elif (b.tooltip in edit.value):
            edit.value[b.tooltip]={'add':'Denied','data':{"time":time.time(),"user":userNameValue.value}}
        elif (b.tooltip not in denied.value.keys()):
            add.value[b.tooltip]={'add':'Denied','data':{"time":time.time(),"user":userNameValue.value}}
        b.style.button_color='red'
        
    def centerImage(b):
        m.location=(imageDict[b.tooltip][2].y,imageDict[b.tooltip][2].x)
    
    #Initialize return GJS and main Image display lists
    groupsGJS=[]
    groupsImg=[]
    
    #Iterate through the entire groups list and add the images and gjs to their respective position
    for group in range(len(groups.keys())):
        print("Adding group #",group," to the web app")
        #Initialize/Reset all relevant lists
        batchesGJS=[]
        batchGJS=[]
        batchesImg=[]
        batchImg=[]
        rowImg=[]
        limit=0
        for key in groups[str(group)]:
            #Limit reached, add the current batch images and gjs to the respective batch list and reset for next batch
            if limit == constants.IMG_SHOWN:
                batchImg.append(HBox(rowImg))
                batchesImg.append(VBox(batchImg))
                batchesGJS.append(batchGJS)
                batchGJS=[]
                batchImg=[]
                rowImg=[]
                limit=0
            
            #Load GJS
            batchGJS.append(folium.GeoJson(ee.Geometry(eval(imageDict[key][1])['geometry']).transform("EPSG:4326",1).getInfo(),name='tile'+str(key)))
            #.add_to(m)
            
            #Add and reset row if row is equal to NUMROWIMG
            if (len(rowImg) >= constants.NUM_IMG_ROW):
                batchImg.append(HBox(rowImg))
                rowImg=[]
                
            #Initialize Buttons and tie their respective click events
            confirmSquareButton = Button(
                description='Confirm Square Kiln',
                tooltip=str(key)
            )
            confirmCircleButton = Button(
                description='Confirm Circle Kiln',
                tooltip=str(key)
            )
            confirmBothButton = Button(
                description='Confirm Both Kilns',
                tooltip=str(key)
            )
            denyButton = Button (
                description="No Kiln",
                tooltip=str(key)
            )
            
            if str(key) in confirmed.value.keys():
                if confirmed.value[str(key)]['kilnType']=='Square':
                    confirmSquareButton.style.button_color='green'
                elif confirmed.value[str(key)]['kilnType']=='Circle':
                    confirmCircleButton.style.button_color='green'
                else:
                    confirmBothButton.style.button_color='green'
                    
            if str(key) in denied.value.keys():
                denyButton.style.button_color='red'
                
            centerButton = Button (
                description="Center to Image",
                tooltip=str(key)
            )

            confirmSquareButton.on_click(functools.partial(confirmKiln,buttons=[confirmCircleButton,confirmBothButton,denyButton],kilnType='Square'))
            confirmCircleButton.on_click(functools.partial(confirmKiln,buttons=[confirmSquareButton,confirmBothButton,denyButton],kilnType='Circle'))
            confirmBothButton.on_click(functools.partial(confirmKiln,buttons=[confirmSquareButton,confirmCircleButton,denyButton],kilnType='Both'))
            denyButton.on_click(functools.partial(denyKiln,buttons=[confirmSquareButton,confirmCircleButton,confirmBothButton]))
            centerButton.on_click(centerImage)    
            
            #Create Output for the matplotlib plots and tie them to the respective image plot
            output=Output(layout={'border': '1px solid black'})
            output2=Output(layout={'border':'1px solid black'})
            with output:
                fig = plt.figure(figsize=(6,6))
                hr=plt.imshow(hrImgDict[key][0])
                plt.show(fig);
            with output2:
                fig2=plt.figure(figsize=(6,6))
                lr=plt.imshow(imageDict[key][0])
                plt.show(fig2);
            
            #Consolidate and order the Images and Buttons into a VBox and increase the limit
            rowImg.append(VBox([
                HBox([
                    output,
                    output2
                ]),
                HBox([
                    VBox([
                        HBox([
                            confirmSquareButton,
                            confirmCircleButton,
                            confirmBothButton
                        ]),
                        HBox([
                            denyButton,
                            centerButton
                        ])
                    ])
                ])
            ]))
            limit+=1
            
        #Do one final append to add any straggling gjs and images and add the total batch set to the respective group list
        batchesGJS.append(batchGJS)
        groupsGJS.append(batchesGJS)
        batchImg.append(HBox(rowImg))
        batchesImg.append(VBox(batchImg))
        groupsImg.append(batchesImg)
    return groupsGJS,groupsImg

def InitiateUI(confirmed,denied,groups,batch,datasetName,constants):
    #Clear existing plts
    plt.clf()
    
    #load image data from storage
    print("Loading Data")
    lrfile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+"LrBatch"+str(batch),'r+b')
    hrfile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+"HrBatch"+str(batch),'r+b')
    imageDict=pkl.load(lrfile)
    hrImgDict=pkl.load(hrfile)
    lrfile.close()
    hrfile.close()
      
    m = folium.Map()
    
    class TextBox:
        def __init__(self,initial='noUser'):
            self.value=initial
        def changeName(self,text):
            self.value=text
            return;
        
    userNameValue=TextBox()
    
    class updateList:
        def __init__(self,initial=[]):
            self.value=initial
            
    confirmedClass=updateList(confirmed)
    deniedClass=updateList(denied)
    
    class editRequest:
        def __init__(self):
            self.value={}
    class addRequest:
        def __init__(self):
            self.value={}
    
    edit=editRequest()
    add=addRequest()
    
    #iterate through each cluster group and pre-load all the UI and return the lists
    print("Initializing UI!")
    groupGJS,groupImg=loadGroup(imageDict,hrImgDict,groups,m,userNameValue,confirmedClass,deniedClass,edit,add,constants)
    print('Done!')
    
    #Initialize counter class and two counters to iteratively keep track between on-click even
        
    class Counter:
        def __init__(self, initial=0):
            self.value = initial

        def increment(self, amount=1):
            self.value += amount
            return self.value

        def reset(self):
            self.value=0

        def __iter__(self, sentinal=False):
            return iter(self.increment, sentinal)

    groupcounter=Counter()
    batchcounter=Counter()
    
    #create buttons and textBox to insert to our layout
    nextGroup=Button(
        description="Next Group",
        tooltip="Proceed"
    )
    
    saveData=Button(
        description="Save Data",
        tooltip="Save"
    )
    
    userName=Text(
        placeholder='Type your name!',
        description='User Name',
        disabled=False
    )
    
    submitName=Button(
        description="Submit",
        tooltip="Submit"
    )
    

    curLabel=Label(value="User: "+userNameValue.value+", Working on: Group: "+str(groupcounter.value)+", Batch: "+str(batchcounter.value))

    
    #On-Click call the next batch of gjs and images or the next group if the batches are done for that group
    def saveDataFunc(confirmed,denied,add,edit,b):
        print('Saving')
        newconfirmedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Confirmed','r+b')
        newdeniedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Denied','r+b')
        newConfirmed=pkl.load(newconfirmedFile)
        newDenied=pkl.load(newdeniedFile)
        newconfirmedFile.close()
        newdeniedFile.close()
        for req in add.value:
            if add.value[req]['add']!='Denied':
                newConfirmed[req]=add.value[req]['data']
            elif add.value[req]['add']=='Denied':
                newDenied[req]=add.value[req]['data']
                
        for req in edit.value:
            if edit.value[req]['add']!='Denied':
                newConfirmed[req]=edit.value[req]['data']
                if req in newDenied:
                    del(newDenied[req])
            elif edit.value[req]['add']=='Denied':
                newDenied[req]=edit.value[req]['data']
                if req in newConfirmed:
                    del(newConfirmed[req])
        newconfirmedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Confirmed','w+b')
        newdeniedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Denied','w+b')
        pkl.dump(newConfirmed,newconfirmedFile)
        pkl.dump(newDenied,newdeniedFile)
        newconfirmedFile.close()
        newdeniedFile.close()
        confirmed.value=newConfirmed
        denied.value=newDenied
        add.value={}
        edit.value={}
        
        
    def submitNameFunc(b,userNameValue=userNameValue,groupcounter=groupcounter,batchcounter=batchcounter):
        bad='1234567890-=[]\\;\',./!@#$%^&*()_+{}|:\"<>? '
        if userName.value and not any(x in userName.value for x in bad):
            print('Setting User to',userName.value)
            userNameValue.changeName(userName.value)
            curLabel.value="User: "+userNameValue.value+", Working on: Group: "+str(groupcounter.value)+", Batch: "+str(batchcounter.value)
            b.style.button_color='green'
        else:
            print('Empty input or not alphabetic found in input')
            b.style.button_color='red'
    
    def loadNextGroup(b,groupcounter=groupcounter,batchcounter=batchcounter):
        if (groupcounter.value < len(groups.keys())):
            #Reset map
            m = folium.Map()
            
            #Check if Batch # is larger than it should be and move onto the next group
            if (batchcounter.value >= ceil(len(groups[str(groupcounter.value)])/constants.IMG_SHOWN)):
                groupcounter.increment()
                batchcounter.reset()
            
            curLabel.value="User: "+userNameValue.value+", Working on: Group: "+str(groupcounter.value)+", Batch: "+str(batchcounter.value)
            curBatch.clear_output()
            with curBatch:
                display(curLabel)
                
            #Print current grouping information for debugging/early termination purposes
            print('group #')
            print(groupcounter.value)
            print('batch #')
            print(batchcounter.value)
            print('gjs group batch #')
            print(len(groupGJS[groupcounter.value]))
            print('group size')
            print(len(groupImg[groupcounter.value]))
            
            #Add the relevant group-batch gjs to the map
            for gjs in groupGJS[groupcounter.value][batchcounter.value]:
                gjs.add_to(m)
            
            mapOutput.clear_output()
            with mapOutput:
                display(m)
            
            #Initialize the group-batch images from the list and display it using a clearable output to effectively 'refresh'
            horizontalBox=VBox([groupImg[groupcounter.value][batchcounter.value]])
            
            HiImg.clear_output()
            with HiImg:
                display(horizontalBox)
            
            #Increase step counter!
            batchcounter.increment()
            
    #Tie next group button to it's on-click event
    saveData.on_click(functools.partial(saveDataFunc,confirmedClass,deniedClass,add,edit))
    nextGroup.on_click(loadNextGroup)
    submitName.on_click(submitNameFunc)

    #create custom box layout for the future image section, map and buttons
    HiImg=Output(layout={'border': '1px solid black'})
    horizontalBox=Box([HiImg],layout=Layout(
        display='flex',
        flex_flow='row',
        width='100%'
    ))
    curBatch=Output(layout={'border': '1px solid black'})
    with curBatch:
        display(curLabel)
    mapOutput=Output(layout={'border':'1px solid black'})
    with mapOutput:
        display(m)
    horizontalBox1=Box([mapOutput,curBatch],layout=Layout(
        display='flex',
        flex_flow='column',
        width='100%'
    ))
    
    #create actual layout for the UI
    maps=VBox([horizontalBox1,horizontalBox])
    button=HBox([nextGroup,saveData,userName,submitName])
    total=VBox([maps,button])
    
    #Return the callable UI!
    return total


def main(group,batch,datasetName,constants):
    #Initialize the basic variables
    if not os.path.exists(constants.GROUPING_ROOT+datasetName+'/'):
        os.mkdir(constants.GROUPING_ROOT+datasetName+'/')
    if not os.path.exists(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Confirmed'):
        confirmed={}
        confirmedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Confirmed','w+b')
        pkl.dump(confirmed,confirmedFile)
        confirmedFile.close()
    else:
        confirmedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Confirmed','r+b')
        confirmed=pkl.load(confirmedFile)
        confirmedFile.close()
    if not os.path.exists(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Denied'):
        denied={}
        deniedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Denied','w+b')    
        pkl.dump(denied,deniedFile)
        deniedFile.close()
    else:
        deniedFile=open(constants.GROUPING_ROOT+datasetName+'/'+datasetName+'Denied','r+b')    
        denied=pkl.load(deniedFile)
        deniedFile.close()
    return InitiateUI(confirmed,denied,group,batch,datasetName,constants)