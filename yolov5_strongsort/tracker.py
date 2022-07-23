import argparse
from csv import writer, DictWriter

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        in_path ='../reference_vids/traffic_30s.avi',
        out_path ='../reference_vids/traffic_out.avi',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_csv=True,  # save results to *.txt
        count_obj=None, # enable obj counting between 2 y coords
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    # Catch invalid vid format
    if Path(in_path).suffix[1:] not in (VID_FORMATS):
        LOGGER.error('Invalid input video format')

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(in_path, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None], [None]

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # Create strongsort object instance
    strongsort_obj = StrongSORT(
            strong_sort_weights,
            device,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )

    outputs = [None]

    # New csv to store results in for analytics
    if save_csv:
        header = ['frame', 'id', 'box_left_x', 'box_top_y', 'box_width', 'box_height']
        with open(save_csv, 'w') as f:
            write_obj = writer(f)
            write_obj.writerow(header)

    # New csv to store counts
    if count_obj:
        y1 = count_obj[0] # Get desired start and stop y coords
        y2 = count_obj[1]
        count = {} # Key is class, value is count
        counted = [] # IDs already counted

    # Run tracking
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None], [None]
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = '../reference_vids/visualize' if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        dets = pred[0] # All dets contained within first val of pred tensor
        seen += 1
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        curr_frames = im0

        s += '%gx%g ' % im.shape[2:]  # print string

        annotator = Annotator(im0, line_width=2, pil=not ascii)
        if cfg.STRONGSORT.ECC:  # camera motion compensation
            strongsort_obj.tracker.camera_update(prev_frames, curr_frames)

        if dets is not None and len(dets):
            # Rescale boxes from img_size to im0 size
            dets[:, :4] = scale_coords(im.shape[2:], dets[:, :4], im0.shape).round()

            # Print results
            for c in dets[:, -1].unique():
                n = (dets[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(dets[:, 0:4])
            confs = dets[:, 4]
            clss = dets[:, 5]

            # pass detections to strongsort
            t4 = time_sync()
            outputs = strongsort_obj.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4

            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    # box dimensions
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]

                    # Write MOT compliant results to file
                    if save_csv:
                        data = [frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h] # MOT format
                        with open(save_csv, 'a') as f:
                            write_obj = writer(f)
                            write_obj.writerow(data)

                    if out_path or show_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        annotator.box_label(bboxes, label, color=colors(c, True))

                    # Count unique tracked objects between two y coords
                    if count_obj:
                        if y1 < bbox_top < y2 and id not in counted: # Between stop and start and not already counted
                            if names[c] in count:
                                count[names[c]] += 1
                            else:
                                count[names[c]] = 1
                            counted.append(id)

            LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

        else:
            LOGGER.info('No detections')

        # Stream results
        im0 = annotator.result()
        if show_vid:
            if count_obj: # Visualise count stop start lines
                cv2.line(im0, (0, y1), (im0.shape[1], y1), (0, 255, 0), 2) # Start count
                cv2.line(im0, (0, y2), (im0.shape[1], y2), (0, 255, 0), 2) # Stop count
            cv2.imshow(str(p), im0)
            
            # quit if q is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("quitting program...")
                return

        # Save results (image with detections)
        if out_path:
            if vid_path != out_path: # if new video
                vid_path = out_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

        prev_frames = curr_frames

    if count_obj:
        header = ['class', 'count']
        with open('count_objs.csv', 'w') as f:
            count_obj = writer(f)
            count_obj.writerow(header)
            for key, value in count.items():
                count_obj.writerow([key, value])

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', type=str, default='../reference_vids/1080p_traffic_2s.mp4', help='input video file path')
    parser.add_argument('-o', '--out_path', type=str, default='../reference_vids/tracker_out.mp4', help='.mp4 output video file path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-csv', help='file path to save csv results to')
    parser.add_argument('--count-obj', nargs='+', type=int, help='count obj classes which pass through two y coords')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)