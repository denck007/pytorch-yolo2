
import numpy as np
import os
import time

class history(object):
    '''
        Class to track the history of model training
        After every save, the data is cleared from memory, but will still hold onto the last self.avgOver
            values. To get all of the data saved to disk self.save(forceAll=True) must be called
            
        After creating the object as his, call it with:
            his(<variable name>,<batch_number>,<variable_value>)
        Get a string of latest values with:str(his)
            Note that if you have not updated all of the values on the current batch,
                the data for the previous batch will be returned
        Get the number of batches that have been entered with len(), same warning as str()
        Get string of all the data in the history with: 
            his.dataToString(allData=True)
    '''
    def __init__(self,outName,avgOver=100,saveEvery=1000,toShow=None):
        '''
            :params: outName: The csv file to save the history to
            :params: avgOver: How many batches to average the data over, if == 0, will average over entire history
            :params: saveEvery: how often to save, if is <0, will never save
            :params: toShow: list, the names of the properties to show when str() is called,
                            None will show all the parameters
        '''
        
        assert saveEvery >avgOver or saveEvery<0, "saveEvery must be greater than the avgOver value!"
        assert avgOver>=0, "avgOver must be greater than 0!"
        self.outName = outName
        self.avgOver = avgOver
        self.saveEvery = saveEvery
        self.toShow = toShow
        self.ds = {} # initialize the data dict
        self.avg = {} # initialize dict that holds the running average
        self.avgable = {} # dict holding the state of if values can/should be averaged
        self.names = [] # list of all the keys in the dicts, so we don't have to constantly call list()

    def __call__(self,epoch,batchnum,data,avg=None):
        '''
        Call the object to update a parameter
            :params: epoch: the current epoch number
            :params: batchnum: the batch number of the data
            :params: data: dict of properties to update
            :params: avg: bool, should the parameter be averaged. the first time each parameter is called,
                            the avg property is checked and recorded. If it is set to None, it is assumed to be true.
        '''
        # Go over the new data, update the history values
        for key in data.keys():
            if key not in self.ds.keys(): #initialize if needed
                self.ds.update({key:np.array([[epoch,batchnum,data[key]]])})
                self.names.append(key)
                if avg is None:
                    avg = True
                self.avgable.update({key:avg})
            else:
                self.ds.update({key:np.append(self.ds[key],[[epoch,batchnum,data[key]]],0)})
        
            # if the data can be averaged, average over the avgOver values
            # Note that when avgOver>ds.shape[0] this still works, but the data may be jumpy
            if self.avgable[key]:
                self.avg.update({key:np.average(self.ds[key][-self.avgOver:,2])})

        # check if we should save
        # The first time there are enough data values, save it and leave the loop
        # The shouldSave parameter is to make sure that when each value is updated in a batch,
        #   it will only save at the end of the batch
        shouldSave = False
        for key in self.ds.keys():
            if self.ds[key].shape[0] > self.saveEvery:
                shouldSave=True
            else:
                shouldSave = False
                break # not enough data in one of the keys, so just escape 
        if shouldSave and self.saveEvery>0:
            self.save()
        
    def save(self,forceAll=False):
        '''
        Save the data
        :params: forceAll: bool, True will save all the data points out, leaving nothing in the history to average over
                            should set to True the last time the save function is called
        '''
        startTime = time.time()
        # make sure all the keys have the same batchnum data
        self._validateBatchNumbers()

        if os.path.isfile(self.outName): # if the file exists modify the headers accordingly
            writeHeaders = False
        else:
            writeHeaders = True

        if forceAll: # check if to save every data point
             outData = self.dataToString(allData=True,writeHeaders=writeHeaders)
        else:
            outData = self.dataToString(allData=False,writeHeaders=writeHeaders)

        with open(self.outName,'a') as f:
            f.write(outData)
        print("Saved history for batches {:.0f} - {:.0f} took {:.3e} seconds".format(self.ds[self.names[0]][0,1],
                                                                                    self.ds[self.names[0]][-self.avgOver,1],
                                                                                    time.time()-startTime))

        # remove the old data from the history
        for key in self.ds.keys():
            if forceAll:
                self.ds[key] = np.array([[]])
            else:
                self.ds[key] = self.ds[key][-self.avgOver:,:]

    def dataToString(self,allData=False,writeHeaders=False):
        '''
        Turn all the data into a string
        Useful for saving and for displaying all the raw history points
        
        :params: allData: bool, if true returns everything in the history buffer as a string
                                if false excludes the last self.avgOver datapoints
        :params: writeHeaders: bool, should the column headers be included in the returned string
        '''
        #self._validateBatchNumbers() # dont need this as long as we call __len__() in the function
        if allData:
            endOffset = self.__len__()
        else:
            endOffset = self.__len__() - self.avgOver

        outString = "\nepoch,batchnum"
        outData = self.ds[self.names[0]][:endOffset,:2].reshape(-1,2)
        for key in self.ds.keys():
            outString += ",{}".format(key)
            print("outData.shape: {}\tself.ds[key][:endOffset,2].reshape(-1,1).shape: {}".format(outData.shape,self.ds[key][:endOffset,2].reshape(-1,1).shape))
            outData = np.concatenate((outData,self.ds[key][:endOffset,2].reshape(-1,1)),axis=1)
        
        if not writeHeaders: # clear out header if it is not to be included
            outString = ""
        for row in range(outData.shape[0]):
            for col in range(outData.shape[1]):
                if col == 0:
                    outString += "\n{:.0f}".format(outData[row,col])
                else:
                    outString += ",{:0.3e}".format(outData[row,col])
        return outString


    def _validateBatchNumbers(self):
        '''
        Verifies that the batch numbers for each of the values are the same
        '''
        for key in self.ds.keys():
            assert np.allclose(self.ds[key][:,1],self.ds[self.names[0]][:,1]), \
                "The batch numbers for each history value do not match! This is not supported!"

    def __len__(self,validateBatchNum=True):
        '''
        Get the number of items in the history arrays
        :params: validateBatchNum: bool, should _validateBatchNumbers() be called? Lets us skip it for just printing out the last datapoint
                                    skipping the validation is dangerous because we may get the incorrect length of the dataset.
                                    Note that this will always be an underestimate of the actual size
        '''
        if validateBatchNum:
            self._validateBatchNumbers()

        length = float('inf')
        for key in self.ds.keys():
            length = min(self.ds[key].shape[0],length)
        return length
    
    def __str__(self):
        '''
            Returns the latest history point as a string
        '''
        # test to see what should be shown, done this way so new keys can be added at any time
        if self.toShow is None:
            valsToShow = list(self.ds.keys())
        else:
            valsToShow = self.toShow            

        numDataPoints = self.__len__(validateBatchNum=False)-1
        first = True
        for key in self.ds.keys():
            if key in valsToShow:
                if first:
                    outData = "batchNum:{:07.0f}".format(self.ds[key][numDataPoints,1])
                    first = False
                if self.avgable[key]:
                    outData += " {}:{:.2e}".format(key,self.avg[key])
                else:
                    outData += " {}:{:.2e}".format(key,self.ds[key][numDataPoints,2])
        return outData

def test():
    a = history("testout.csv",avgOver=2,saveEvery=10)

    for ii in range(20):
        a(epoch=0,batchnum=ii,data={"v1":ii/2,"v2":ii*2})
    a.save(forceAll=True)
if __name__ is "__main__":
    test()
