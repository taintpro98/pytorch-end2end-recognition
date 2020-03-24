import numpy as np
import cv2
import torch

def collate_fn(batch):
    image_names, img, score_map, geo_map, training_mask, transcripts, boxes = zip(*batch)
    bs = len(score_map)
    images = []
    score_maps = []
    geo_maps = []
    training_masks = []

    for i in range(bs):
        if img[i] is not None:
            a = torch.from_numpy(img[i])
            a = a.permute(2, 0, 1)
            images.append(a)
            b = torch.from_numpy(score_map[i])
            b = b.permute(2, 0, 1)
            score_maps.append(b)
            c = torch.from_numpy(geo_map[i])
            c = c.permute(2, 0, 1)
            geo_maps.append(c)
            d = torch.from_numpy(training_mask[i])
            d = d.permute(2, 0, 1)
            training_masks.append(d)


    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    mapping = []
    texts = []
    bboxs = []
    for index, gt in enumerate(zip(transcripts, boxes)):
        for t, b in zip(gt[0], gt[1]):
            mapping.append(index)
            texts.append(t)
            bboxs.append(b)

    mapping = np.array(mapping)
    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis = 1).astype(np.float32)
    image_names = [p for p in image_names]

    return image_names, images, score_maps, geo_maps, training_masks, texts, bboxs, mapping

def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly

def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype = np.uint8)
    score_map = np.zeros((h, w), dtype = np.uint8)
    geo_map = np.zeros((h, w, 4), dtype = np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype = np.uint8)

    ############
    rectangles = []
    for poly in polys:
        rectangles.append(np.array([
            poly[0][0], poly[0][1], poly[1][0] - poly[0][0], poly[3][1] - poly[0][1]
        ]))
    rectangles = np.stack(rectangles, axis=0)
    ############

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        
        cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        
        p0, p1, p2, p3 = poly
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype = np.float32)
            # top
            geo_map[y, x, 0] = y - p0[1]
            # right
            geo_map[y, x, 1] = p1[0] - x
            # down
            geo_map[y, x, 2] = p2[1] - y
            # left
            geo_map[y, x, 3] = x - p3[0]
    
    return score_map, geo_map, training_mask, rectangles