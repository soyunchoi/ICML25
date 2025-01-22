# pylint: disable=W0212
"""
Webcam demo application

Implementation adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/webcam.py

"""

import time
import logging

import torch
import matplotlib.pyplot as plt
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

import openpifpaf
from openpifpaf import decoder, network, visualizer, show, logger
from openpifpaf import datasets

from ..visuals import Printer
from ..network import Loco, preprocess_pifpaf, load_calibration
from ..predict import download_checkpoints

LOG = logging.getLogger(__name__)

def factory_from_args(args):

    # Model
    dic_models = download_checkpoints(args)
    args.checkpoint = dic_models['keypoints']

    logger.configure(args, LOG)  # logger first

    assert len(args.output_types) == 1 and 'json' not in args.output_types

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults
    if not args.output_types:
        args.output_types = ['multi']

    args.figure_width = 10
    args.dpi_factor = 1.0

    args.z_max = 10
    args.show_all = True
    args.no_save = True
    args.batch_size = 1

    if args.long_edge is None:
        args.long_edge = 144
    # Make default pifpaf argument
    args.force_complete_pose = True
    LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args, dic_models


def webcam(args):

    assert args.mode in 'mono'
    assert cv2

    args, dic_models = factory_from_args(args)

    # Load Models
    net = Loco(model=dic_models[args.mode], mode=args.mode, device=args.device,
               n_dropout=args.n_dropout, p_dropout=args.dropout)

    # for openpifpaf predicitons
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # Start recording
    import pyrealsense2 as rs
    import numpy as np
    #import cv2
    #pipeline = rs.pipeline()
    #config = rs.config()
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    #profile = pipeline.start(config)
    #device = profile.get_device()
    #device.hardware_reset()
    
    #cam = cv2.namedWindow('Depth Stream', cv2.WINDOW_NORMAL)
    
    #cam = cv2.VideoCapture(args.camera)
    cam = cv2.VideoCapture(args.camera)
    visualizer_mono = None

    while True:
        start = time.time()
        ret, frame = cam.read()
        scale = (args.long_edge)/frame.shape[0]
        image = cv2.resize(frame, None, fx=scale, fy=scale)
        height, width, _ = image.shape
        LOG.debug('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        data = datasets.PilImageList(
            [pil_image], preprocess=predictor.preprocess)

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False,
            pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

        for (_, _, _) in data_loader:

            for idx, (preds, _, _) in enumerate(predictor.dataset(data)):
		
                if idx == 0:
                    pifpaf_outs = {
                        'pred': preds,
                        'left': [ann.json_data() for ann in preds],
                        'image': image}

        if not ret:
            break
        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC pressed
            LOG.info("Escape hit, closing...")
            break

        kk = load_calibration(args.calibration, pil_image.size, focal_length=args.focal_length)
        boxes, keypoints = preprocess_pifpaf(
            pifpaf_outs['left'], (width, height))
            
        #print(boxes)
        dic_out = net.forward(keypoints, kk)
        dic_out = net.post_process(dic_out, boxes, keypoints, kk)

        if 'social_distance' in args.activities:
            dic_out = net.social_distance(dic_out, args)
        if 'raise_hand' in args.activities:
            dic_out = net.raising_hand(dic_out, keypoints)
        if visualizer_mono is None:  # it is, at the beginning
            visualizer_mono = Visualizer(kk, args)(pil_image)  # create it with the first image
            visualizer_mono.send(None)

        LOG.debug(dic_out)
        visualizer_mono.send((pil_image, dic_out, pifpaf_outs))
        
        #print(dic_out)
        
        #print("AAAAAAAAAAAAAAAAAAAAA")
        #print(boxes[0][0])
        #print(boxes[idx][2] - boxes[idx][0])
        
        
        #if idx > 0:
        	#print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        	#print(boxes[idx][2] - boxes[idx][0])

        #index = np.array([1])
        #for i in idx:
        	#print(boxes[idx][2] - boxes[idx][0])
        
        #print(dic_out["boxes"])
        #print(pil_image)
        #print(pifpaf_outs)
        
        
        #print(pifpaf_outs)
        #print('AAAAAAAAAAAAAAAAAAAAAAA')
        #print(dic_out)

        end = time.time()
        LOG.info("run-time: {:.2f} ms".format((end-start)*1000))
        
        t_n = (end-start)*1000
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        #print(boxes[idx][2] - boxes[idx][0])
        #width = boxes[idx][2] - boxes[idx][0]
        #print(width)
        
        if t_n < 20:
        	x = np.full((5, 2),100)
        	np.save('location', x)
        	
        
        
        
        
        n_h = len(dic_out["dds_pred"])
        w1 = {}
        index = 0
        xy = []
        #if n_h == 0:
        	#print("AAAAAAAAAAAAAAAAAAAAAA")
        while index < n_h:
            w1 = boxes[index][2] - boxes[index][0]
            #print(dic_out['angles'][index])
            a1 = dic_out['angles'][index]
            #a1 = angles[index]
            index += 1
        	#print(w1)
            if 120 <= w1 <= 150:
            	d = 0.35
            if 100 <= w1 < 120:
            	d = 0.5
            elif 80<= w1 < 100:
            	d = 1
            elif 70<= w1 < 80:
            	d = 2
            elif 60<= w1 < 70:
            	d = 3
            elif 40<= w1 < 60:
            	d = 4
            elif 20<= w1 < 40:
            	d = 5            	
            elif w1 < 20:
            	d = 7            	
            else:
            	d = 0.2           	
            	
            #elif w1 < 30:
        	#d = 5
            #else:
        	#d = 0.2
            x = d * np.cos(a1)
            y = d * np.sin(a1)
            xy.append([x, y])
            
        if len(xy) == 0:
                x = np.full((5, 2), 15)
                np.save('location', x)
        else:
                np.save('location', np.array(xy))
        # new_matrix = [index, dic_out['angles'][index], dic_out['action'][index]]

    cam.release()

    cv2.destroyAllWindows()


class Visualizer:
    def __init__(self, kk, args):
        self.kk = kk
        self.args = args

    def __call__(self, first_image, fig_width=1.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width *
                                 first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, output_path="",
                          kk=self.kk, args=self.args)

        figures, axes = printer.factory_axes(None)

        for fig in figures:
            fig.show()

        while True:
            image, dic_out, pifpaf_outs = yield

            # Clears previous annotations between frames
            axes[0].patches = []
            axes[0].lines = []
            axes[0].texts = []
            if len(axes) > 1:
                axes[1].patches = []
                axes[1].lines = [axes[1].lines[0], axes[1].lines[1]]
                axes[1].texts = []

            if dic_out and dic_out['dds_pred']:
                printer._process_results(dic_out)
                printer.draw(figures, axes, image, dic_out, pifpaf_outs['left'])
                mypause(0.01)


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)
