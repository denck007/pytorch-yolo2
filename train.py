from __future__ import print_function
import sys
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
import os
from utils import bbox_iou,nms,get_region_boxes,load_class_names,read_data_cfg,file_lines,logging
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet

from history import history


# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]
weightfile    = sys.argv[3]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = data_options['train']
testlist      = data_options['valid']
trainLabelFolder = data_options['trainLabelFolder']
testLabelFolder = data_options['testLabelFolder']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])
print("num_workers: {}".format(10))

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = int(max_batches*batch_size/nsamples+1)
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 2  # epoches
dot_interval  = 70  # batches

trainHistoryFname = "historyTrain.csv"
testHistoryFname = "historyTest.csv"
saveHistoryEvery = 5000
averageHistoryOver = 100 # only used for the command line output, does not affect the output file

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

if not os.path.exists(backupdir):
    os.mkdir(backupdir)
    
###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()

region_loss.seen  = model.seen
processed_batches = model.seen/batch_size

init_width        = model.width
init_height       = model.height
init_epoch        = int(model.seen/nsamples )

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, testLabelFolder, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=True), # does not work if set to False
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    toShow = ["loss","lossX","lossY","lossW","lossH","lossConf","lossCls","totalTime"]
    trainHistory = history(trainHistoryFname,avgOver=averageHistoryOver,saveEvery=saveHistoryEvery,toShow=toShow)

    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, trainLabelFolder, shape=(init_width, init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), 
                       train=True, 
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    t1 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()
            #target= target.cuda() #this gets put on gpu later in region_loss, that way we only have 1 copy of it there
        t3 = time.time()
        data, target = Variable(data), Variable(target); t4 = time.time()
        optimizer.zero_grad(); t5 = time.time()
        output = model(data); t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss, lossData = region_loss(output, target); t7 = time.time()
        loss.backward(); t8 = time.time()
        optimizer.step(); t9 = time.time()

        lossData.update({"loadDataTime":(t2-t1),
                        "cpuToCudaTime":t3-t2,
                        "cudaToVariableTime":t4-t3,
                        "zeroGradTime":t5-t4,
                        "forwardFeatureTime":t6-t5,
                        "forwardLossTime":t7-t6,
                        "backwardTime":t8-t7,
                        "stepTime":t9-t8,
                        "totalTime":t9-t1})

        trainHistory(epoch,batch_idx,lossData) # log the history

        t1 = time.time() # reset at end to get data loading time
        if batch_idx%50==0:
            print(trainHistory)

    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))
    
    trainHistory.save(forceAll=True) # save all the remaining data in the history monitor
    

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
    toShow = ["loss","precision","recall","fscore"]
    testHistory = history(testHistoryFname,avgOver=0,saveEvery=-1,toShow=toShow)

    #model.eval() # does not work in eval mode
    model.train()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors

    totalOverall = 0.0
    proposalsOverall = 0.0
    correctOverall = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        total = 0.0 # reset these every batch
        proposals = 0.0
        correct = 0.0

        if use_cuda:
            data = data.cuda()
        data = Variable(data)#, volatile=True)
        
        output = model(data)
        _, lossData = region_loss(output, Variable(target)) # don't need the loss values for anything, just the misc data
        optimizer.zero_grad() #remove the gradient from the optimizer
        output = output.data # get the data so we can work on it here

        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            

            #assert False
            # This is the function to draw the resulting boxes on the image
            #draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            
            
            
            
            
            
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
            total = total + num_gts
    
            for j in range(len(boxes)):
                if boxes[j][4] > conf_thresh:
                    proposals = proposals+1

            for k in range(num_gts):
                box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1
        

        totalOverall += total; proposalsOverall += proposals; correctOverall += correct
        precision = 1.0*correct/(proposals+eps); recall = 1.0*correct/(total+eps); fscore = 2.0*precision*recall/(precision+recall+eps)
        lossData.update({"correctBBoxes":correct,
                        "proposedBBoxes":proposals,
                        "groundTruthBBoxes":num_gts,
                        "totalBBoxes":total,
                        "precision":precision,
                        "recall":recall,
                        "fscore":fscore})
        testHistory(epoch,batch_idx,lossData) # log the history
        if batch_idx%100==0:
            print(testHistory)

    precision = 1.0*correctOverall/(proposalsOverall+eps)
    recall = 1.0*correctOverall/(totalOverall+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("\nprecision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    testHistory.save(forceAll=True)

evaluate =False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    print("init_epoch: {} max_epochs: {}".format(init_epoch,max_epochs))
    for epoch in range(init_epoch, max_epochs): 

        train(epoch)
        print("Finished training on epoch {}".format(epoch))
        test(epoch)
        
        # save the last weights
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))