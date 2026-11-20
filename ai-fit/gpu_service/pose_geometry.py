"""Pixel-space pose anchors and a conservative output pose check.

These are geometric controls for the inpainting mask, not a trained pose adapter.
"""
import cv2
import numpy as np


def visible(point, minimum=.5):
    values=(point.x, point.y, point.visibility, point.presence)
    return bool(np.isfinite(values).all() and point.visibility >= minimum and point.presence >= minimum
                and 0 <= point.x <= 1 and 0 <= point.y <= 1)


def cropped_upper(points, size):
    """Recognise an observed upper torso cut by the bottom of the image.

    Extrapolated hips identify the crop only; callers must not use their
    coordinates as anchors or create pixels outside the source photograph.
    """
    if points is None or len(points)!=33 or not all(visible(points[i],.65) for i in (0,11,12)):
        return False
    if all(visible(points[i],.4) for i in (23,24)):
        return False
    hips=np.array([(points[i].x,points[i].y) for i in (23,24)])
    if not np.isfinite(hips).all() or hips[:,1].min()<.95 or hips[:,1].max()<=1:
        return False
    width,height=size
    shoulder_y=(points[11].y+points[12].y)*height/2
    span=np.hypot((points[11].x-points[12].x)*width,(points[11].y-points[12].y)*height)
    return bool(.15*height<shoulder_y<.65*height and height-shoulder_y>=max(.25*height,span*.55))


def torso_anchors(points, size):
    width,height=size
    if points is None or len(points)!=33 or not all(visible(points[i],.5) for i in (11,12)):
        return None
    shoulder=np.array([(points[11].x+points[12].x)*width/2,(points[11].y+points[12].y)*height/2])
    if all(visible(points[i],.5) for i in (23,24)):
        hip=np.array([(points[23].x+points[24].x)*width/2,(points[23].y+points[24].y)*height/2])
    elif cropped_upper(points,size):
        # This is a frame boundary, not an inferred hip/body measurement.
        hip=np.array([shoulder[0],height-1.])
    else:
        return None
    return shoulder,hip


def align_upper_shoulders(image, before, after, neck_anchor='mouth'):
    """Correct a small generated shoulder drop with fixed mouth/hip anchors.

    Only a bounded vertical resampling is permitted. Large anatomy changes
    remain for the output validator to reject. Callers must reapply the source
    edit mask and protected pixels after this operation.
    """
    from PIL import Image
    anchor=(0,) if neck_anchor=='nose' else (9,10)
    if (before is None or after is None or len(before)!=33 or len(after)!=33
            or not all(visible(p[i],.5) for p in (before,after) for i in (*anchor,11,12))):
        return image
    h,w=image.height,image.width
    geometry=torso_anchors(before,image.size)
    if geometry is None or torso_anchors(after,image.size) is None:return image
    cropped=not all(visible(before[i],.5) for i in (23,24))
    mouth=sum(before[i].y for i in anchor)*h/len(anchor)
    source_shoulder=sum(before[i].y for i in (11,12))*h/2
    generated_shoulder=sum(after[i].y for i in (11,12))*h/2
    hip=(h-2.) if cropped else geometry[1][1]
    torso=hip-source_shoulder
    neck=source_shoulder-mouth
    drop=generated_shoulder-source_shoulder
    if (torso<=0 or neck<=torso*.03 or not torso*.015 < drop <= torso*(.16 if cropped else .12)
            or not .08 < drop/neck <= .40 or not 0 < mouth < source_shoulder < generated_shoulder < hip < h-1):
        return image
    # Inverse mapping: a destination shoulder pixel samples the generated
    # shoulder. Mouth, hips and the rest of the photo retain their coordinates.
    rows=np.arange(h,dtype=np.float32)
    source_rows=np.interp(rows,[0,mouth,source_shoulder,hip,h-1],
                               [0,mouth,generated_shoulder,hip,h-1]).astype(np.float32)
    map_y=np.broadcast_to(source_rows[:,None],(h,w)).copy()
    map_x=np.broadcast_to(np.arange(w,dtype=np.float32)[None,:],(h,w)).copy()
    return Image.fromarray(cv2.remap(np.asarray(image),map_x,map_y,cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE))


def neck_protection(points, size, labels):
    """Keep visible source neck skin above the shoulder line at source coordinates."""
    width,height=size
    result=np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33 or not all(visible(points[i],.5) for i in (11,12)):
        return result
    shoulder_x=(points[11].x+points[12].x)*width/2
    shoulder_y=(points[11].y+points[12].y)*height/2
    span=np.hypot((points[11].x-points[12].x)*width,(points[11].y-points[12].y)*height)
    rows,columns=np.ogrid[:height,:width]
    neck=(np.abs(columns-shoulder_x)<span*.22)&(rows<shoulder_y+span*.04)
    # Parsing sometimes calls neck skin an arm, so preserve all skin classes
    # within this anatomically bounded region, without retaining shirt fabric.
    result[neck & np.isin(labels,(13,14,15))]=255
    return result


def body_width_difference(before, after, points, size):
    """Compare torso silhouette widths without inferring body measurements."""
    width,height=size
    anchors=torso_anchors(points,size)
    if anchors is None:
        return {'passed': True, 'available': False, 'ratios': []}
    source=np.asarray(before.resize(size),dtype=np.float32)>.5
    output=np.asarray(after.resize(size),dtype=np.float32)>.5
    shoulder,hip=anchors
    shoulder_y,hip_y=shoulder[1],hip[1]
    ratios=[]
    band=max(1,round((hip_y-shoulder_y)*.015))
    for fraction in (.35,.50,.65,.80):
        row=round(shoulder_y+(hip_y-shoulder_y)*fraction)
        if not 0 <= row < height:
            continue
        center=round(shoulder[0]*(1-fraction)+hip[0]*fraction)
        widths=[]
        for mask in (source,output):
            section=[]
            for y in range(max(0,row-band),min(height,row+band+1)):
                line=mask[y]
                if not 0 <= center < width or not line[center]:
                    section.append(0)
                    continue
                left=right=center
                while left>0 and line[left-1]:left-=1
                while right+1<width and line[right+1]:right+=1
                section.append(right-left+1)
            widths.append(float(np.median(section)))
        if widths[0]>=10:
            ratios.append(widths[1]/widths[0])
    # Multiple sections must agree before rejecting a narrower silhouette.
    # A single row can be affected by crossed arms or a garment opening.
    narrowed=sum(ratio<.82 for ratio in ratios)
    available=len(ratios)>=3
    return {'passed': available and narrowed<2, 'available': available, 'ratios': ratios}


def hand_protection(points, size, *, protect_wrist=False, clip_cuff=False):
    width, height = size
    result=np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33:
        return result
    shoulder_span=np.hypot((points[11].x-points[12].x)*width,(points[11].y-points[12].y)*height)
    if not np.isfinite(shoulder_span):
        return result
    radius=max(2,round(shoulder_span*.035))
    for elbow,indices in ((13,(15,17,19,21)),(14,(16,18,20,22))):
        # Do not paint a guessed hand when only an occluded wrist is available.
        if not visible(points[indices[0]]):
            continue
        coords=[(round(points[i].x*(width-1)),round(points[i].y*(height-1))) for i in indices if visible(points[i])]
        hand=np.zeros_like(result)
        if protect_wrist and visible(points[elbow],.55):
            # Retain the wrist joint and nearby accessories at the cuff.
            wrist_radius = max(radius,round(shoulder_span*.065))
            cv2.circle(hand,coords[0],wrist_radius,255,-1)
        if len(coords)>=2:
            shape=np.zeros_like(result)
            cv2.fillConvexPoly(shape,cv2.convexHull(np.array(coords,dtype=np.int32)),255)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
            hand=np.maximum(hand,cv2.dilate(shape,kernel))
        if clip_cuff and visible(points[elbow],.55):
            # Keep the hand side of the wrist, rather than a circle extending
            # into old sleeves that the parser may misclassify as skin.
            wx,wy=coords[0]
            dx=wx-points[elbow].x*(width-1)
            dy=wy-points[elbow].y*(height-1)
            length=np.hypot(dx,dy)
            if length>=12:
                rows,columns=np.ogrid[:height,:width]
                hand[(columns-wx)*dx+(rows-wy)*dy < 0]=0
        result=np.maximum(result,hand)
    return result


def pose_difference(before, after, size, category, neck_anchor='mouth'):
    """No alignment: source and output must retain coordinates in the same image.

    Thresholds are conservative engineering checks, not calibrated probabilities.
    """
    width,height=size
    if len(before)!=33 or len(after)!=33:
        raise ValueError('Pose comparison requires 33 landmarks.')
    anchors=torso_anchors(before,size) if category=='upper' else None
    if anchors is None:
        shoulder=np.array([(before[11].x+before[12].x)*width/2,(before[11].y+before[12].y)*height/2])
        hip=np.array([(before[23].x+before[24].x)*width/2,(before[23].y+before[24].y)*height/2])
    else:
        shoulder,hip=anchors
    scale=max(float(np.linalg.norm(hip-shoulder)),1.)
    indices=[11,12,13,14,15,16,23,24]
    if category in ('lower','overall'):
        indices += [25,26,27,28]
    movement={};missing=[]
    for i in indices:
        if not visible(before[i], .65 if i in (25,26,27,28) else .5):
            continue
        if not visible(after[i],.4):
            missing.append(i);continue
        movement[i]=float(np.hypot((before[i].x-after[i].x)*width,(before[i].y-after[i].y)*height)/scale)
    shifted=[i for i,v in movement.items() if v > (.22 if i in (13,14,15,16,27,28) else .28)]
    neck_changed=False
    neck_ratio=None
    anchor=(0,) if neck_anchor=='nose' else (9,10)
    if all(visible(p[i],.5) for p in (before,after) for i in (*anchor,11,12)):
        before_mouth=sum(before[i].y for i in anchor)*height/len(anchor)
        after_mouth=sum(after[i].y for i in anchor)*height/len(anchor)
        original_length=shoulder[1]-before_mouth
        output_length=(after[11].y+after[12].y)*height/2-after_mouth
        if original_length>scale*.03:
            neck_ratio=float(output_length/original_length)
            # Even a modest shoulder drop makes an unchanged face look like
            # it has a longer neck. Allow a small landmark noise margin only.
            neck_changed=output_length-original_length>max(scale*.025,original_length*.12)
    return {'passed':not missing and not shifted and not neck_changed,'missing':missing,
            'shifted':shifted,'displacement':movement,'neck_changed':bool(neck_changed),
            'neck_ratio':neck_ratio}


def leg_corridor(points, size):
    """Bound semantic-mask repair to the subject's lower legs, not scene people."""
    width,height=size;mask=np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33:return mask
    span=abs(points[11].x-points[12].x)*width
    for knee,ankle in ((25,27),(26,28)):
        if not visible(points[knee],.4) or not visible(points[ankle],.4):continue
        start=(round(points[knee].x*(width-1)),round(points[knee].y*(height-1)))
        end=(round(points[ankle].x*(width-1)),round(points[ankle].y*(height-1)))
        cv2.line(mask,start,end,255,max(3,round(span*.40)))
    return mask


def foot_protection(points, size, labels):
    """Preserve footwear edges without retaining misclassified trouser legs."""
    width,height=size
    shoes=np.isin(labels,(18,19)).astype(np.uint8)*255
    span=width*.25
    if points is not None and len(points)==33 and visible(points[11],.4) and visible(points[12],.4):
        span=abs(points[11].x-points[12].x)*width
    radius=max(2,round(span*.07))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    result=cv2.dilate(shoes,kernel)
    if points is None or len(points)!=33:return result
    for heel,toe in ((29,31),(30,32)):
        if not visible(points[heel],.4) or not visible(points[toe],.65):continue
        a=(round(points[heel].x*(width-1)),round(points[heel].y*(height-1)))
        b=(round(points[toe].x*(width-1)),round(points[toe].y*(height-1)))
        cv2.line(result,a,b,255,max(3,radius*2))
    return result


def upper_arm_editable(points, size):
    """Allow the sleeve cap to change around reliable shoulder landmarks."""
    width,height = size
    mask = np.zeros((height,width),dtype=np.uint8)
    if len(points) != 33 or not all(visible(points[i], .65) for i in (11,12)):
        return mask
    span = abs(points[11].x-points[12].x)*width
    for shoulder, elbow in ((11,13),(12,14)):
        a,b = points[shoulder],points[elbow]
        if not visible(b,.65):
            continue
        start = (round(a.x*(width-1)),round(a.y*(height-1)))
        end = (round((a.x+.35*(b.x-a.x))*(width-1)),
               round((a.y+.35*(b.y-a.y))*(height-1)))
        cv2.line(mask,start,end,255,max(3,round(span*.20)))
    return mask


def sleeve_corridor(points, size, *, long_sleeves=True):
    """Make the visible arm available to short- and long-sleeve inpainting."""
    width,height = size
    mask = np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33 or not all(visible(points[i],.55) for i in (11,12)):
        return mask
    span = np.hypot((points[11].x-points[12].x)*width,
                    (points[11].y-points[12].y)*height)
    for indices in ((11,13,15),(12,14,16)):
        shoulder,elbow,wrist = (points[i] for i in indices)
        if not long_sleeves:
            if visible(shoulder,.4) and visible(elbow,.4):
                start = (round(shoulder.x*(width-1)),round(shoulder.y*(height-1)))
                end = (round((shoulder.x+.6*(elbow.x-shoulder.x))*(width-1)),
                       round((shoulder.y+.6*(elbow.y-shoulder.y))*(height-1)))
                cv2.line(mask,start,end,255,max(3,round(span*.30)))
            continue
        for a,b,width_fraction in ((shoulder,elbow,.52),(elbow,wrist,.42)):
            if visible(a,.40) and visible(b,.40):
                start = (round(a.x*(width-1)),round(a.y*(height-1)))
                end = (round(b.x*(width-1)),round(b.y*(height-1)))
                cv2.line(mask,start,end,255,max(3,round(span*width_fraction)))
    return mask
