'''
Testing the onnx lib with pytorch branchynet early exit model.

- Saving a model to onnx
- Running a loaded onnx model against same pytorch model


- Check what the differences are with diff batch sizes
- Loading a model from pytorch and changing to onnx
'''

#importing pytorch models to test
from models.msdnet import MSDNet
import args2
#from args import main
from args2 import path_check

#args file:
import os
import glob
import time
import argparse


from tools import *

import torch
import torch.nn as nn
import torch.optim as optim

#to get inputs for testing
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt
import argparse

#to test onnx methods
import io
import torch.onnx

import onnx
import onnxruntime
print('1')
def to_onnx(model, input_size, batch_size=1,
        path='outputs/onnx', fname='brn.onnx', test_in=None):
    #convert the model to onnx format - trial with onnx lib
    #if speedy:
    #    fname = 'speedy-'+name
    #    model.set_fast_inf_mode()
    #else:
    #    fname = 'slow-'+name
    #    model.eval()

    sv_pnt = os.path.join(path, fname)
    if not os.path.exists(path):
        os.makedirs(path)

    if test_in is None:
        x = torch.randn(batch_size, *input_size)
    else:
        x=test_in

    #trying the scripty thing
    scr_model = torch.jit.script(model)
    print("PRINTING PYTORCH MODEL SCRIPT")
    print(scr_model.graph, "\n")
    ex_out = scr_model(x) # get output of script model

    torch.onnx.export(
        scr_model,      # model being run
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        example_outputs=ex_out,
        input_names = ['input'],   # the model's input names
        output_names = ['exit'],#, 'eeF'], # the model's output names
        #dynamic_axes={#'input' : {0 : 'batch_size'}, # NOTE not used, variable length axes
                      #'exit' : {0 : 'exit_size'}#,
                      #'eeF' : {0 : 'exit_size'}
                      #}
    )
    return sv_pnt

#'/home/localadmin/phd/earlyexitnet/outputs/pre_Trn_bb_2021-07-09_141616/pretrn-joint-8-2021-07-09_142311.pth'
#'brn-top1ee-bsf-lessOps-trained.onnx'
print('2')
def brn_main(md_pth, save_name):

    print("Running BranchyNet Test")
    bs = 1
    shape = [1,28,28]
    #set up model
    #model = Branchynet(fast_inf_batch_size=bs, exit_threshold=0.1)
    model = MSDNet(args2)

    checkpoint = torch.load(md_pth)
    model.load_state_dict(checkpoint['state_dict'])
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'

    #fast inf pytorch
    model.set_fast_inf_mode()
    print("Finished loading model parameters")

    #generate input
    test_x = torch.ones(1, *shape)
    #torch.randn(1, *shape)

    #pull real input example from MNIST data set
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])
    mnist_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=1, drop_last=True, shuffle=False)

    mnistiter = iter(mnist_dl)
    xb, yb = mnistiter.next() #MNIST example stored in xb

    combi = torch.cat((test_x, test_x, xb), 0) #NOTE currently not used, combines into batch
    #print(combi)


    print("STARTING RUN OF PYTORCH MODEL WITH INPUTS")
    output = model(xb)
    output2 = model(test_x)
    print("PT OUT:", output)
    print("PT OUT2:", output2)


    #save to onnx
    print("SAVING MODEL TO ONNX: ", save_name)
    save_path = to_onnx(model, shape, batch_size=bs, fname=save_name, test_in=xb)
    print("SAVED TO: ",save_path)

    #load from onnx
    print("IMPORTING MODEL FROM ONNX")
    #onnx_model = onnx.load(save_path)
    #onnx.checker.check_model(onnx_model) #running model checker
    #TODO add more onnx checks
    #print("IMPORTED")

    #onnx runtime model

    ort_session = onnxruntime.InferenceSession(save_path) #start onnx runtime session
    def to_numpy(tensor): #function to convert tensor to numpy format
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    print("RUNNING ONNX")
    # compute ONNX Runtime (ort) output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xb)}
    ort_outs = ort_session.run(None, ort_inputs)

    print("ONNX_OUT", ort_outs)
    ort_inputs2 = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
    ort_outs2 = ort_session.run(None, ort_inputs2)

    print("ONNX_OUT2", ort_outs2)


    # compare ONNX Runtime and PyTorch results
    #outlist=[]
    #for out in output:
        #olist=[]
        #for o in out:
        #    if isinstance(o, torch.Tensor):
        #        olist.append(to_numpy(o))
        #outlist.append(olist)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output),
        ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(output2),
        ort_outs2[0], rtol=1e-03, atol=1e-05)
print('2b')

model_names = ['msdnet']
print('5')


def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)
print('3')

def main():
	arg_parser = argparse.ArgumentParser(
		        description='Image classification PK main script')

	exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
	exp_group.add_argument('--save_name', type=str,
		#default='save/default-{}'.format(time.time()),metavar='SAVE',
		               help='path to the experiment logging directory'
		               '(default: save/debug)')
#	exp_group.add_argument('--resume', action='store_true',
#		               help='path to latest checkpoint (default: none)')
#	exp_group.add_argument('--evalmode', default=None,
#		               choices=['anytime', 'dynamic'],
#		               help='which mode to evaluate')
	exp_group.add_argument('--trained_path', type=path_check,
		#type=str, metavar='PATH', default=None,
		               help='path to saved checkpoint (default: none)')
#	exp_group.add_argument('--print-freq', '-p', default=10, type=int,
#		               metavar='N', help='print frequency (default: 100)')
	exp_group.add_argument('--model', choices=['brn','lenet'],
		               help='choose model')
#	exp_group.add_argument('--seed', default=0, type=int,
#		               help='random seed')
#	exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

	
	print('6')
	#brn_main(md_pth=args.trained_path, save_name=args.save_name)
	#args1 = arg_parser.parse_args()
	args = arg_parser.parse_args()
	brn_main(md_pth=args.trained_path, save_name=args.save_name)
		
print('7')
	#args = arg_parser.parse_args()	




























#print('7')
#print('4')
if __name__ == "__main__":
    main()
