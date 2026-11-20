"""CPU clothing segmentation and a garment-focused inference crop."""
import cv2
import numpy as np
import torch
import hashlib
import time
from collections import OrderedDict
from PIL import Image, ImageFilter
from model.SCHP import SCHP
from pose_geometry import hand_protection, leg_corridor, foot_protection, upper_arm_editable, sleeve_corridor, visible, neck_protection, torso_anchors, cropped_upper


def long_sleeve_target(garment):
    if garment is not None and isinstance(garment.info.get('long_sleeves'),bool):
        return garment.info['long_sleeves']
    silhouette=garment.info.get('garment_silhouette') if garment is not None else None
    if silhouette is None:
        return None
    alpha=np.asarray(silhouette)>127
    ys,xs=np.where(alpha)
    if not xs.size:
        return None
    shape=alpha[ys.min():ys.max()+1,xs.min():xs.max()+1]
    sides=max(1,round(shape.shape[1]*.10))
    wings=np.any(shape[:,:sides],axis=1)|np.any(shape[:,-sides:],axis=1)
    rows=np.where(wings)[0]
    return bool(not rows.size or rows.max()>=shape.shape[0]*.60)


def short_upper_target(garment):
    """Recognise a clearly short-bodied product; ambiguous lengths stay unknown."""
    silhouette=garment.info.get('garment_silhouette') if garment is not None else None
    if silhouette is None:
        return False
    alpha=np.asarray(silhouette)>127
    ys,xs=np.where(alpha)
    if not xs.size:
        return False
    shape=alpha[ys.min():ys.max()+1,xs.min():xs.max()+1]
    h,w=shape.shape
    widths=[]
    for y in range(round(h*.55),round(h*.85)):
        if not shape[y,w//2]:
            continue
        left=right=w//2
        while left>0 and shape[y,left-1]:left-=1
        while right+1<w and shape[y,right+1]:right+=1
        widths.append(right-left+1)
    return bool(len(widths)>=max(1,round(h*.15)) and h/np.median(widths)<=1.5)


class ClothingMask:
    def short_source_upper(self, image, landmarks, garment):
        """Give long sleeves more sampling when replacing an existing short top."""
        if long_sleeve_target(garment) is not True or landmarks is None or len(landmarks)!=33:
            return False
        labels=self._labels(image)
        if np.mean(np.isin(labels,(5,6,7,10,11)))<.01:
            return False
        if not all(visible(landmarks[i],.65) and 0<=landmarks[i].x<=1 and 0<=landmarks[i].y<=1 for i in (11,12,13,14)):
            return True
        height,width=labels.shape;radius=max(2,round(min(width,height)*.008))
        for shoulder,elbow in ((11,13),(12,14)):
            a,b=landmarks[shoulder],landmarks[elbow]
            x=round((a.x+(b.x-a.x)*.85)*(width-1));y=round((a.y+(b.y-a.y)*.85)*(height-1))
            patch=labels[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
            if not patch.size or np.mean(np.isin(patch,(5,6,7,10,11)))<.65:
                return True
        return False

    def begin_request(self):
        # Only the GPU admission lock's owner uses this bounded, private cache.
        # Content keys also detect in-place PIL edits; identity keys would not.
        self._request_labels=OrderedDict()
        self.label_stats={'calls':0,'hits':0,'seconds':0.0}

    def end_request(self):
        self._request_labels=None

    def prefer_preserved_upper(self, image, landmarks, garment):
        """Conservative alternate for plain coloured sleeves already covering elbows.

        Printed references and bare/short-sleeved source arms retain the default
        renderer; the alternate has not preserved those reliably in live checks.
        """
        if long_sleeve_target(garment) is not True:
            return False
        silhouette=garment.info.get('garment_silhouette')
        if silhouette is None or landmarks is None or len(landmarks)!=33:
            return False
        alpha=cv2.erode((np.asarray(silhouette)>127).astype(np.uint8),np.ones((5,5),np.uint8))>0
        hsv=cv2.cvtColor(np.asarray(garment.convert('RGB')),cv2.COLOR_RGB2HSV)[alpha]
        if len(hsv)<100:
            return False
        low,high=np.percentile(hsv,(5,95),axis=0)
        # Stay away from neutral fabric, contrasting prints and multi-colour
        # references, where a colour statistic cannot establish design fidelity.
        if low[1]<45 or high[0]-low[0]>12 or high[2]-low[2]>85:
            return False
        contrast=((hsv[:,1]<30)&(hsv[:,2]>max(80,float(np.median(hsv[:,2]))+25)))|(hsv[:,2]>high[2]+45)
        if np.mean(contrast)>.008:
            return False
        if not all(visible(landmarks[i],.65) and 0<=landmarks[i].x<=1 and 0<=landmarks[i].y<=1 for i in (11,12,13,14)):
            return False
        labels=self._labels(image);height,width=labels.shape
        radius=max(2,round(min(width,height)*.008))
        for shoulder,elbow in ((11,13),(12,14)):
            covered=0
            for fraction in (.7,.85,.95):
                a,b=landmarks[shoulder],landmarks[elbow]
                x=round((a.x+(b.x-a.x)*fraction)*(width-1))
                y=round((a.y+(b.y-a.y)*fraction)*(height-1))
                patch=labels[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
                covered+=bool(patch.size and np.mean(np.isin(patch,(5,6,7,10,11)))>.65)
            if covered<2:
                return False
        return True

    def long_upper_region(self, landmarks, size, foreground):
        width,height=size;empty=np.zeros((height,width),dtype=bool)
        if foreground is None or landmarks is None or len(landmarks)!=33 or not all(visible(landmarks[i],.5) for i in (11,12,23,24)):
            return empty,None
        sy=(landmarks[11].y+landmarks[12].y)*height/2
        hy=(landmarks[23].y+landmarks[24].y)*height/2;torso=hy-sy
        if torso<=0:return empty,None
        hem=hy+torso*.8
        if all(visible(landmarks[i],.5) for i in (25,26)):
            knees=(landmarks[25].y+landmarks[26].y)*height/2
            if knees>hy+torso*.3:hem=knees-torso*.05
        hem=min(height-1,round(hem))
        span=np.hypot((landmarks[11].x-landmarks[12].x)*width,(landmarks[11].y-landmarks[12].y)*height)
        cx=(landmarks[23].x+landmarks[24].x)*width/2
        zone=np.zeros((height,width),np.uint8)
        cv2.rectangle(zone,(max(0,round(cx-span*.6)),max(0,round(hy-torso*.35))),
                      (min(width-1,round(cx+span*.6)),hem),1,-1)
        observed=(zone>0)&(np.asarray(foreground)>.45)
        ys,xs=np.where(observed)
        if xs.size<32:return empty,None
        hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
        result=np.zeros_like(zone);cv2.fillConvexPoly(result,hull,1)
        return (result>0)&(zone>0),hem

    def check_upper_length(self, output, landmarks, upper_length, edit_mask, independent_clothing=None):
        if upper_length!='long' or landmarks is None or len(landmarks)!=33 or not all(visible(landmarks[i],.5) for i in (11,12,23,24)):
            return {'passed':True,'available':False}
        height=output.height;hy=(landmarks[23].y+landmarks[24].y)*height/2
        torso=hy-(landmarks[11].y+landmarks[12].y)*height/2
        rows=np.arange(height)[:,None]
        span=np.hypot((landmarks[11].x-landmarks[12].x)*output.width,
                      (landmarks[11].y-landmarks[12].y)*height)
        center=(landmarks[23].x+landmarks[24].x)*output.width/2
        # "Long" includes thigh-length coats; it does not promise a measured
        # knee-length fit. Check visible extension beyond the hips centrally,
        # where sleeves cannot masquerade as a longer body.
        region=((np.asarray(edit_mask)>127)&(rows>hy+torso*.04)&(rows<hy+torso*.18)
                &(np.abs(np.arange(output.width)[None,:]-center)<span*.3))
        if region.sum()<64:return {'passed':True,'available':False}
        cloth=np.isin(self._labels(output),(5,6,7,10,11))
        if independent_clothing is not None:cloth |= independent_clothing
        coverage=float(np.mean(cloth[region]))
        return {'passed':coverage>=.6,'available':True,'coverage':coverage}

    def has_layered_upper(self, image):
        labels=self._labels(image)
        upper=np.count_nonzero(np.isin(labels,(5,6,7,10,11)))
        return bool(upper and np.count_nonzero(labels==7)>upper*.08 and np.count_nonzero(labels==5)>upper*.08)

    def _cropped_arms(self, labels, landmarks):
        region=np.zeros(labels.shape,dtype=np.uint8)
        size=(labels.shape[1],labels.shape[0])
        if not cropped_upper(landmarks,size):return region>0
        # The visible source arms reach the frame boundary and both hands are
        # outside it. Cover only observed arm pixels, never guessed wrists.
        if not all(np.isfinite([landmarks[i].x,landmarks[i].y]).all() and landmarks[i].y>1 for i in (15,16)):
            return region>0
        count,components,stats,_=cv2.connectedComponentsWithStats(np.isin(labels,(14,15)).astype(np.uint8),8)
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]>=labels.shape[0]-2:
                region[components==i]=1
        return region>0

    def __init__(self, checkpoint):
        self.parser = SCHP(str(checkpoint), device='cpu')

    def _labels(self, image):
        # The Windows SCHP PIL path expects BGR.
        cache=getattr(self,'_request_labels',None)
        key=None
        if cache is not None:
            key=(image.mode,image.size,hashlib.sha256(image.tobytes()).digest())
            if key in cache:
                self.label_stats['hits']+=1
                cache.move_to_end(key)
                return cache[key]
        started=time.perf_counter()
        bgr = Image.fromarray(np.asarray(image)[:, :, ::-1].copy())
        with torch.inference_mode():
            labels=np.asarray(self.parser(bgr))
        if cache is not None:
            self.label_stats['calls']+=1
            self.label_stats['seconds']+=time.perf_counter()-started
            cache[key]=labels
            if len(cache)>8:cache.popitem(last=False)
        return labels

    def neck_anchor(self, original, landmarks):
        """Use a visible facial anchor, rather than a guessed mouth in a helmet."""
        if landmarks is None or len(landmarks)!=33:
            return 'mouth'
        labels=self._labels(original)
        radius=max(2,round(min(original.size)*.004))
        def skin_at(index):
            p=landmarks[index]
            if not visible(p,.5):return False
            x,y=round(p.x*(original.width-1)),round(p.y*(original.height-1))
            patch=labels[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
            return bool(patch.size and np.mean(patch==13)>.5)
        return 'mouth' if skin_at(9) and skin_at(10) else 'nose' if skin_at(0) else 'mouth'

    def check_upper_surface(self, preview, landmarks, garment, edit_mask):
        """Reject a large invented light underlayer in a closed dark reference.

        This conservative check is unavailable for light, printed or open-front
        references. It is not a general garment similarity/accuracy score.
        """
        unavailable={'passed':True,'available':False}
        silhouette=garment.info.get('garment_silhouette') if garment is not None else None
        if silhouette is None or landmarks is None or len(landmarks)!=33:
            return unavailable
        anchors=torso_anchors(landmarks,preview.size)
        if anchors is None:
            return unavailable
        alpha=np.asarray(silhouette)>127
        pixels=np.asarray(garment.convert('RGB'))
        if alpha.shape!=pixels.shape[:2] or not alpha.any():return unavailable
        hsv=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV)
        fabric=hsv[alpha]
        if np.mean(fabric[:,2]<160)<.85 or np.mean((fabric[:,2]>180)&(fabric[:,1]<76))>.02:
            return unavailable
        ys,xs=np.where(alpha)
        shape=alpha[ys.min():ys.max()+1,xs.min():xs.max()+1]
        center=shape[round(shape.shape[0]*.35):round(shape.shape[0]*.90),shape.shape[1]//2]
        if not center.size or np.mean(center)<.90:return unavailable
        width,height=preview.size
        shoulder,hip=anchors
        sy,hy=shoulder[1],hip[1]
        if hy<=sy:return unavailable
        span=np.hypot((landmarks[11].x-landmarks[12].x)*width,
                       (landmarks[11].y-landmarks[12].y)*height)
        rows,columns=np.ogrid[:height,:width]
        fraction=(rows-sy)/(hy-sy)
        shoulder_x,hip_x=shoulder[0],hip[0]
        mid=shoulder_x+fraction*(hip_x-shoulder_x)
        region=(fraction>.35)&(fraction<1.0)&(np.abs(columns-mid)<span*.18)
        region &= np.asarray(edit_mask)>96
        if region.sum()<64:return unavailable
        output=cv2.cvtColor(np.asarray(preview),cv2.COLOR_RGB2HSV)
        light_limit=max(110,min(180,float(np.percentile(fabric[:,2],90))+45))
        mismatch=np.ones(output.shape[:2],dtype=bool)
        if np.median(fabric[:,1])>=50:
            chromatic=fabric[fabric[:,1]>50,0]
            dominant=int(np.argmax(np.bincount(chromatic,minlength=180)))
            delta=np.abs(output[:,:,0].astype(np.int16)-dominant)
            mismatch=(output[:,:,1]<25)|(np.minimum(delta,180-delta)>18)
        # A lighter blue/green fold is still the reference colour. Brightness
        # and saturation alone must not mistake it for a neutral inner layer.
        unexpected=(output[:,:,2]>light_limit)&(output[:,:,1]<76)&mismatch
        if np.median(fabric[:,1])>=70:
            # A grey knit underlayer can be much darker than a white vest.
            # Compare it with the closed coloured reference's saturation too.
            unexpected |= ((output[:,:,2]>max(75,float(np.percentile(fabric[:,2],90))+15))
                           &(output[:,:,1]<min(60,float(np.median(fabric[:,1]))*.65))&mismatch)
        unexpected &= region
        ratio=float(unexpected.sum()/region.sum())
        return {'passed':ratio<=.12,'available':True,'unexpected_light_fraction':ratio}

    def restore_background_edges(self, original, preview, foreground, edit_mask, preview_foreground=None):
        """Replace invented edge halos using only source-background context.

        Where the old garment occluded the scene, a local inpainted clean plate
        supplies a narrow edge band. Generated garment, skin and protected pixels
        are never selected by this background-only correction.
        """
        if foreground is None:
            return preview
        source_fg=np.asarray(foreground)>.25
        editable=np.asarray(edit_mask)>96
        background=self._labels(preview)==0
        if preview_foreground is not None:
            # White cloth is sometimes labelled background by SCHP. Require
            # independent output segmentation to agree before removing pixels.
            background &= np.asarray(preview_foreground)<.35
        # Keep a two-pixel safety border away from generated fabric/skin.
        background=cv2.erode(background.astype(np.uint8),np.ones((3,3),np.uint8))>0
        visible_scene=background & editable & ~source_fg
        unknown=background & editable & source_fg
        distance=cv2.distanceTransform(source_fg.astype(np.uint8),cv2.DIST_L2,3)
        unknown &= distance < max(4,min(original.size)*.06)
        if not visible_scene.any() and not unknown.any():
            return preview
        pixels=np.asarray(preview).copy()
        source=np.asarray(original)
        pixels[visible_scene]=source[visible_scene]
        if unknown.any():
            scale=min(1.,768/max(original.size))
            size=(max(1,round(original.width*scale)),max(1,round(original.height*scale)))
            small=np.asarray(original.resize(size,Image.Resampling.LANCZOS))
            hole=cv2.resize(source_fg.astype(np.uint8)*255,size,interpolation=cv2.INTER_NEAREST)
            hole=cv2.dilate(hole,np.ones((5,5),np.uint8))
            plate=cv2.inpaint(small,hole,3,cv2.INPAINT_TELEA)
            plate=cv2.resize(plate,original.size,interpolation=cv2.INTER_LINEAR)
            pixels[unknown]=plate[unknown]
        return Image.fromarray(pixels)

    def prepare_sleeve_refinement(self, preview, landmarks, wrists, size, edit_mask, protected):
        """Refine failed sleeves while the accepted torso stays conditioning."""
        if landmarks is None or len(landmarks)!=33:
            return None
        width,height=preview.size
        if not all(visible(landmarks[i],.5) for i in (11,12)):
            return None
        span=np.hypot((landmarks[11].x-landmarks[12].x)*width,
                      (landmarks[11].y-landmarks[12].y)*height)
        repair=np.zeros((height,width),np.uint8)
        for shoulder,elbow,wrist in ((11,13,15),(12,14,16)):
            if wrist not in wrists or not all(visible(landmarks[i],.55) for i in (elbow,wrist)):
                continue
            s,e,w=(np.array([landmarks[i].x*(width-1),landmarks[i].y*(height-1)])
                   for i in (shoulder,elbow,wrist))
            start=s+(e-s)*.55
            cv2.line(repair,tuple(np.round(start).astype(int)),tuple(np.round(e).astype(int)),255,max(5,round(span*.48)))
            cv2.line(repair,tuple(np.round(e).astype(int)),tuple(np.round(w).astype(int)),255,max(5,round(span*.40)))
        repair[(np.asarray(edit_mask)<96)|(np.asarray(protected)>0)]=0
        repair[hand_protection(landmarks,preview.size,protect_wrist=True,clip_cuff=True)>0]=0
        ys,xs=np.where(repair>0)
        if not xs.size:return None
        # Retain the new torso, collar and face as fixed context for sleeve fill.
        context=[landmarks[i] for i in (0,11,12,23,24) if visible(landmarks[i],.5)]
        left=min(xs.min(),*(p.x*width for p in context))-span*.20
        right=max(xs.max()+1,*(p.x*width for p in context))+span*.20
        top=min(ys.min(),*(p.y*height for p in context))-span*.20
        bottom=max(ys.max()+1,*(p.y*height for p in context))+span*.12
        crop_w=max(right-left,(bottom-top)*size[0]/size[1])
        crop_h=max(bottom-top,crop_w*size[1]/size[0])
        left=(left+right-crop_w)/2;top=(top+bottom-crop_h)/2
        if crop_w<=width:left=max(0,min(left,width-crop_w))
        if crop_h<=height:top=max(0,min(top,height-crop_h))
        box=tuple(round(v) for v in (left,top,left+crop_w,top+crop_h))
        crop=Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'white')
        crop.paste(preview,(-box[0],-box[1]))
        mask=Image.fromarray(repair).crop(box).resize(size,Image.Resampling.NEAREST)
        return crop.resize(size,Image.Resampling.LANCZOS),mask,box

    def check_sleeve_coverage(self, output, landmarks, garment, source=None, edit_mask=None):
        """Check both long-to-short and short-to-long sleeve drift."""
        expected=long_sleeve_target(garment)
        if expected is None or landmarks is None or len(landmarks)!=33:
            return {'passed':True,'available':False,'short_arms':[]}
        labels=self._labels(output)
        short_arms=[]
        long_arms=[]
        checked=0
        for elbow,wrist in ((13,15),(14,16)):
            if not visible(landmarks[elbow],.55) or not visible(landmarks[wrist],.55):
                continue
            a,b=landmarks[elbow],landmarks[wrist]
            length=np.hypot((a.x-b.x)*output.width,(a.y-b.y)*output.height)
            if length<12:
                continue
            checked+=1
            exposed=0
            clothed=0
            for fraction in (.35,.50,.65,.75):
                x=round((a.x+fraction*(b.x-a.x))*(output.width-1))
                y=round((a.y+fraction*(b.y-a.y))*(output.height-1))
                patch=labels[max(0,y-2):min(output.height,y+3),max(0,x-2):min(output.width,x+3)]
                if patch.size and np.mean(np.isin(patch,(14,15)))>.6:
                    exposed+=1
                if patch.size and np.mean(np.isin(patch,(5,6,7,10,11)))>.6:
                    clothed+=1
            if expected and exposed>=2:
                short_arms.append(wrist)
            if not expected and clothed>=2:
                long_arms.append(wrist)
        if expected and source is not None and edit_mask is not None:
            region=self._cropped_arms(self._labels(source),landmarks)&(np.asarray(edit_mask)>127)
            if region.sum()>=32:
                checked+=1
                if np.mean(np.isin(labels[region],(14,15)))>.15:
                    short_arms.append(15)
        return {'passed':not short_arms and not long_arms,'available':checked>0,
                'short_arms':short_arms,'long_arms':long_arms}

    def restore_plain_surface(self, preview, garment, edit_mask=None, landmarks=None):
        """Repair small contrasting emblems only on confirmed plain neutral cloth."""
        silhouette=garment.info.get('garment_silhouette') if garment is not None else None
        if silhouette is None:return preview
        alpha=np.asarray(silhouette)>127
        sample=np.asarray(garment.convert('RGB'))
        if alpha.shape!=sample.shape[:2] or not alpha.any():return preview
        fabric=sample[alpha]
        luma=fabric.mean(axis=1)
        hsv=cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
        if np.median(hsv[:,:,1][alpha])>45 or np.mean((hsv[:,:,1][alpha]>70)&(hsv[:,:,2][alpha]>80))>.0005:
            return preview
        white=luma.mean()>210
        dark=luma.mean()<80
        if not white and not dark:return preview
        # Centre closures are intentional. Any off-centre reference contrast
        # disables this correction, so actual prints/brands are retained.
        ys,xs=np.where(alpha);center=(xs.min()+xs.max())/2
        reference_gray=cv2.cvtColor(sample,cv2.COLOR_RGB2GRAY)
        contrast=(reference_gray<85) if white else (reference_gray>110)
        reference_interior=cv2.distanceTransform(alpha.astype(np.uint8),cv2.DIST_L2,3)>max(2,min(garment.size)*.005)
        contrast &= reference_interior & (np.abs(np.arange(sample.shape[1])[None,:]-center)>(xs.max()-xs.min()+1)*.05)
        if np.count_nonzero(contrast)>max(2,alpha.sum()*.0005):return preview
        labels=self._labels(preview)
        clothing=np.isin(labels,(5,6,7,10,11)).astype(np.uint8)
        gray=cv2.cvtColor(np.asarray(preview),cv2.COLOR_RGB2GRAY)
        local=cv2.medianBlur(gray,31)
        interior=cv2.distanceTransform(clothing,cv2.DIST_L2,3)>max(4,min(preview.size)*.01)
        if edit_mask is not None:
            interior &= np.asarray(edit_mask.resize(preview.size))>127
        cy,cx=np.where(clothing)
        if not cx.size:return preview
        center=(cx.min()+cx.max())/2;span=cx.max()-cx.min()+1
        if landmarks is not None and len(landmarks)==33 and all(visible(landmarks[i],.55) for i in (11,12)):
            center=(landmarks[11].x+landmarks[12].x)*preview.width/2
            span=abs(landmarks[11].x-landmarks[12].x)*preview.width
        interior &= np.abs(np.arange(preview.width)[None,:]-center)>span*.05
        marks=(((gray<85)&(local>165) if white else (gray>110)&(local<90)&(gray.astype(np.int16)-local>40))&interior).astype(np.uint8)
        # Group adjacent letters first; repairing isolated bright strokes leaves
        # their grey outlines behind and copies them into the repaired pixels.
        marks=cv2.morphologyEx(marks,cv2.MORPH_CLOSE,np.ones((3,7),np.uint8))
        count,components,stats,_=cv2.connectedComponentsWithStats(marks,8)
        remove=np.zeros_like(marks)
        for index in range(1,count):
            x,y,w,h,area=stats[index]
            if 3<=area and w*h<=clothing.sum()*.015 and .25<=w/max(h,1)<=4:
                remove[max(0,y-2):y+h+2,max(0,x-2):x+w+2]=255
        if not remove.any():return preview
        remove=cv2.dilate(remove,np.ones((3,3),np.uint8))
        remove[~interior]=0
        return Image.fromarray(cv2.inpaint(np.asarray(preview),remove,3,cv2.INPAINT_TELEA))

    def restore_exposed_skin(self, original, preview, edit_mask=None, garment=None):
        """Keep the source tone where the preview still shows exposed skin."""
        source_labels = self._labels(original)
        original_skin = np.isin(source_labels, (13, 14, 15))
        source_clothing = np.isin(source_labels,(5,6,7,10,11)).astype(np.uint8)
        ys,xs = np.where(source_clothing)
        if xs.size:
            radius=max(2,round((xs.max()-xs.min()+1)*.025))
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
            # Dark cuffs can be parsed as skin. Do not paste source pixels
            # back onto the new sleeve within this narrow clothing edge band.
            near_clothing=cv2.dilate(source_clothing,kernel)>0
            original_skin &= ~near_clothing | (source_labels==13)
        visible_skin = np.isin(self._labels(preview), (13, 14, 15)).astype(np.uint8)
        # Keep garment boundary pixels from the preview so new cuffs and
        # collars do not inherit skin from under the shirt.
        visible_skin = cv2.erode(visible_skin, np.ones((5, 5),dtype=np.uint8))
        shared_skin = (original_skin & (visible_skin>0)).astype(np.uint8)
        if edit_mask is not None and long_sleeve_target(garment) is True:
            # Dark new fabric can be labelled as arm skin. Re-pasting the bare
            # source forearm makes holes in an otherwise complete sleeve.
            # Actual hands/neck are already protected by the source edit mask.
            shared_skin[np.isin(source_labels,(14,15))&(np.asarray(edit_mask)>96)]=0
        distance = cv2.distanceTransform(shared_skin, cv2.DIST_L2, 3)
        # Feather inward so the skin correction does not create a cutout
        # seam at the wrist or bleed into adjacent cloth.
        mask = Image.fromarray(np.clip(distance*(255/5),0,255).astype(np.uint8))
        return Image.composite(original,preview,mask)

    def restore_lower_layers(self, original, preview, landmarks, edit_mask, confirmed_lower=None):
        """A long upper garment may cover trousers, but cannot replace them.

        One-piece conditioning supplies garment length. Its independently
        generated trousers/legs/background below the waist stay source pixels.
        """
        if confirmed_lower is None or landmarks is None or len(landmarks)!=33 or not all(visible(landmarks[i],.5) for i in (11,12,23,24)):
            return preview
        hy=(landmarks[23].y+landmarks[24].y)*preview.height/2
        torso=hy-(landmarks[11].y+landmarks[12].y)*preview.height/2
        lower=np.arange(preview.height)[:,None]>hy-torso*.35
        labels=self._labels(preview)
        # Pale/floral coats are often labelled background or legs by SCHP.
        # Only two-parser agreement on trousers can restore lower source cloth.
        restore=lower&(np.asarray(edit_mask)>96)&np.isin(labels,(9,12,16,17,18,19))&confirmed_lower
        return Image.composite(original,preview,Image.fromarray(restore.astype(np.uint8)*255))

    def garment_only(self, image, category):
        """Remove donor skin/head/background from a detected worn-garment image."""
        rgb=image.convert('RGB')
        with torch.inference_mode():
            labels=np.asarray(self.parser(Image.fromarray(np.asarray(rgb)[:,:,::-1].copy())))
        targets={'upper':(5,6,7,10,11), 'lower':(9,12), 'overall':(5,6,7,9,10,11,12)}[category]
        keep=np.isin(labels,targets)
        if np.count_nonzero(keep)<labels.size*.01:
            raise ValueError('Could not isolate the outfit from its model. Use a garment-only product photo.')
        face_fraction=float(np.mean(labels==13))
        other_person_fraction=float(np.mean(np.isin(labels,(2,14,15))))
        if face_fraction<.002 or (other_person_fraction<.002 and face_fraction<.015):
            # Pose can hallucinate a person in a flat hoodie or T-shirt. Do
            # not cut off its hood/sleeves unless the human parser confirms
            # a face with additional person evidence (or a clear large face).
            return image
        pixels=np.asarray(rgb).copy();pixels[~keep]=255
        ys,xs=np.where(keep);pad=max(3,round((xs.max()-xs.min()+1)*.04))
        box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(rgb.width,int(xs.max())+pad+1),min(rgb.height,int(ys.max())+pad+1))
        result=Image.fromarray(pixels).crop(box)
        # Preserve sleeve-shape metadata after removing a donor model. Without
        # it, a worn T-shirt becomes an unknown long-sleeve target downstream.
        result.info['garment_silhouette']=Image.fromarray(keep.astype(np.uint8)*255).crop(box)
        if category=='upper':
            # A worn short-sleeved shirt can have a wide lower torso, making
            # its cropped silhouette look like long sleeves. Large, elongated
            # exposed arm regions establish short sleeves independently of
            # that silhouette. Small hands at long cuffs do not qualify.
            for label in (14,15):
                arm=(labels==label).astype(np.uint8)
                count,components,stats,_=cv2.connectedComponentsWithStats(arm,8)
                if any(stats[i,cv2.CC_STAT_AREA]>labels.size*.008
                       and stats[i,cv2.CC_STAT_HEIGHT]>labels.shape[0]*.18
                       for i in range(1,count)):
                    result.info['long_sleeves']=False
                    break
        return result

    def prepare(self, person, category, size=(768, 1024), landmarks=None, include_protection=False, foreground=None, garment=None, upper_length='standard'):
        labels = self._labels(person)
        silhouette = garment.info.get('garment_silhouette') if garment is not None else None
        long_sleeves = True
        short_bottoms = False
        if silhouette is not None:
            alpha = np.asarray(silhouette) > 127
            ys, xs = np.where(alpha)
            if xs.size:
                x0,y0,x1,y1 = xs.min(),ys.min(),xs.max()+1,ys.max()+1
                shape = alpha[y0:y1,x0:x1]
                if category == 'upper':
                    # The sleeve silhouette reaches the sides below the chest
                    # on long-sleeved product photos. Unknown shapes retain the
                    # existing long-sleeve path instead of guessing joint points.
                    long_sleeves = long_sleeve_target(garment)
                elif category == 'lower':
                    short_bottoms = shape.shape[0] < shape.shape[1]*1.05
        targets = {'upper': (5, 6, 7, 10, 11), 'lower': (9, 12, 16, 17),
                   'overall': (5, 6, 7, 9, 10, 11, 12, 16, 17)}[category]
        clothing = np.isin(labels, targets).astype(np.uint8) * 255
        bare_torso = np.zeros_like(labels,dtype=bool)
        bare_collar_top=None
        if (category == 'upper' and np.count_nonzero(clothing) < labels.size*.005
                and foreground is not None and landmarks is not None and len(landmarks)==33
                and all(visible(landmarks[i],.65) for i in (11,12,23,24))):
            # A shirtless source has no old shirt to segment. Build a bounded
            # torso only when pose, skin labels and foreground independently
            # agree; never substitute a large unverified rectangle.
            polygon=np.array([(round(landmarks[i].x*(person.width-1)),
                               round(landmarks[i].y*(person.height-1)))
                              for i in (11,12,24,23)],dtype=np.int32)
            center=(polygon[0]+polygon[1])/2
            span=float(np.linalg.norm(polygon[0]-polygon[1]))
            collar=np.array([[center[0]-span*.14,center[1]-span*.16],
                             [center[0]+span*.14,center[1]-span*.16]],dtype=np.int32)
            polygon=np.concatenate((polygon,collar))
            region=np.zeros_like(clothing)
            cv2.fillConvexPoly(region,cv2.convexHull(polygon),1)
            torso=region>0
            # SCHP has no dedicated bare-chest class and can assign background
            # there. Permit that label only inside the pose torso and the
            # independently confident person foreground.
            skin=np.isin(labels,(0,13,14,15))
            agreement=torso & skin & (np.asarray(foreground)>.8)
            if torso.sum() and agreement.sum()/torso.sum()>.7:
                bare_torso=agreement
                clothing[bare_torso]=255
                if all(visible(landmarks[i],.65) for i in (9,10)):
                    mouth_y=(landmarks[9].y+landmarks[10].y)*person.height/2
                    shoulder_y=(landmarks[11].y+landmarks[12].y)*person.height/2
                    if mouth_y<shoulder_y:
                        bare_collar_top=round(mouth_y+(shoulder_y-mouth_y)*.4)
                        rows,columns=np.ogrid[:person.height,:person.width]
                        # Bare neck/chest is often parsed as background. The
                        # old narrow polygon left skin wedges over a new collar.
                        collar=((rows>=bare_collar_top)&(rows<shoulder_y+span*.1)
                                &(np.abs(columns-center[0])<span*.32)
                                &np.isin(labels,(0,14,15))&(np.asarray(foreground)>.8))
                        bare_torso |= collar
                        clothing[collar]=255
        repair=np.zeros_like(labels,dtype=bool)
        if category in ('lower','overall') and foreground is not None:
            # SCHP can label pale trousers as background. Repair only pixels
            # independently marked as this person's foreground and near legs.
            repair=(np.asarray(foreground)>.8)&(leg_corridor(landmarks,person.size)>0)&(labels==0)
            clothing[repair]=255
        if np.count_nonzero(clothing) < labels.size * .005:
            raise ValueError('Could not identify the clothing to replace. Use a clear front-facing photo.')
        hem_limit=None
        long_overlay=np.zeros(labels.shape,dtype=bool)
        if category=='upper' and upper_length=='long':
            long_overlay,hem_limit=self.long_upper_region(landmarks,person.size,foreground)
            clothing[long_overlay]=255
            if hem_limit is not None:
                below=np.arange(person.height)[:,None]>hem_limit
                clothing[below&(sleeve_corridor(landmarks,person.size)==0)]=0
        if (category=='upper' and upper_length!='long' and short_upper_target(garment) and landmarks is not None
                and len(landmarks)==33 and all(visible(landmarks[i],.5) for i in (11,12,23,24))):
            shoulders=(landmarks[11].y+landmarks[12].y)*person.height/2
            hips=(landmarks[23].y+landmarks[24].y)*person.height/2
            torso=hips-shoulders
            if torso>0:
                # A short shirt replaces the upper region of a dress/kurta,
                # leaving its lower portion as the existing bottom layer.
                rows=np.where(clothing>0)[0]
                if rows.size and rows.max()>hips+torso*.7:
                    hem_limit=round(hips+torso*.15)
                    below=np.arange(person.height)[:,None]>hem_limit
                    arm_region=sleeve_corridor(landmarks,person.size)>0
                    clothing[below & ~arm_region]=0
        ys, xs = np.where(clothing > 0)
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        garment_width = x1-x0
        radius = max(7, round(garment_width * .05))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        hem_band = np.zeros_like(clothing, dtype=bool)
        if category == 'upper':
            # A button-up's split hem must not become a permanent opening in
            # a replacement T-shirt. Permit a continuous hem in this narrow band.
            hem_top = max(y0, y1-round(garment_width*.15))
            hem_band[hem_top:min(person.height, y1+radius+1)] = True
            hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
            filled = np.zeros_like(clothing)
            cv2.fillConvexPoly(filled, hull, 255)
            clothing[hem_band] = np.maximum(clothing[hem_band], filled[hem_band])
        mask = cv2.dilate(clothing, kernel)
        edge_repair=np.zeros_like(labels,dtype=bool)
        if category == 'overall' and foreground is not None:
            # Human parsing can miss pale fabric at the silhouette. Permit the
            # existing narrow dilation only where person segmentation agrees.
            edge_repair=(mask>0)&(np.asarray(foreground)>.5)&(labels==0)
        # Keep the face, hair and unrelated clothes out of the editable area.
        protected = np.isin(labels, (1, 2, 4, 13, 18, 19))
        if category=='upper' and landmarks is not None and len(landmarks)==33 and all(visible(landmarks[i],.65) for i in (11,12)):
            # Coarse hair parsing includes a thin fringe of the old shirt.
            # Permit that fringe to change only beside recognised clothing,
            # below the shoulders; keep head hair and the long-hair core fixed.
            hair=(labels==2).astype(np.uint8)
            fringe_radius=max(1,round(min(person.size)*.004))
            hair_core=cv2.erode(hair,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*fringe_radius+1,2*fringe_radius+1)))
            shoulder_y=(landmarks[11].y+landmarks[12].y)*person.height/2
            fringe=(hair>0)&(hair_core==0)&(np.arange(person.height)[:,None]>shoulder_y)&(mask>0)
            if hair_core.any():
                hsv=cv2.cvtColor(np.asarray(person.convert('RGB')),cv2.COLOR_RGB2HSV)
                sample=hsv[hair_core>0]
                saturation=float(np.median(sample[:,1]))
                if saturation>=80:
                    # Light neutral coat pixels occasionally form a broader
                    # false hair fringe. Remove only colour outliers near that
                    # boundary; blonde/grey hair does not enable this rule.
                    distance=cv2.distanceTransform(hair,cv2.DIST_L2,3)
                    fringe |= ((hair>0)&(distance<max(2,min(person.size)*.02))
                               &(hsv[:,:,1]<saturation*.5)&(hsv[:,:,2]>float(np.median(sample[:,2]))+30)
                               &(np.arange(person.height)[:,None]>shoulder_y)&(mask>0))
            protected[fringe]=False
        protected[bare_torso & (labels!=13)]=False
        neck=neck_protection(landmarks,person.size,labels)>0
        if bare_torso.any():
            shoulder_y=(landmarks[11].y+landmarks[12].y)*person.height/2
            shoulder_span=np.hypot((landmarks[11].x-landmarks[12].x)*person.width,
                                   (landmarks[11].y-landmarks[12].y)*person.height)
            # Leave room for a new collar at the base of a bare neck while
            # retaining the upper neck and the original face.
            neck &= np.arange(person.height)[:,None] < shoulder_y-shoulder_span*.12
        protected |= neck
        if bare_collar_top is not None:
            # Keep the original upper neck/face fixed even after mask dilation.
            rows,columns=np.ogrid[:person.height,:person.width]
            protected |= (rows<bare_collar_top)&(np.abs(columns-center[0])<span*.32)
        if category == 'upper':
            protected |= np.isin(labels, (9, 12)) & ~hem_band & ~long_overlay
            protected |= np.isin(labels, (16, 17)) & ~long_overlay
            arms = np.isin(labels, (14, 15)) & ~bare_torso
            sleeves = arms & (sleeve_corridor(landmarks,person.size,long_sleeves=long_sleeves)>0)
            cropped_arms=np.zeros(labels.shape,dtype=bool)
            if long_sleeves is True and foreground is not None:
                candidate_arms=self._cropped_arms(labels,landmarks)
                if candidate_arms.any() and np.mean(np.asarray(foreground)[candidate_arms])>.6:
                    cropped_arms=candidate_arms
                sleeves |= cropped_arms
            if bare_torso.any() and long_sleeves and foreground is not None:
                # The same missing bare-chest class can leave bare arms labelled
                # background. Permit their independently segmented foreground
                # only inside the pose corridor, so full sleeves reach wrists.
                sleeves |= ((labels==0) & (np.asarray(foreground)>.8)
                            & (sleeve_corridor(landmarks,person.size)>0))
            cuff_radius=max(2,round(garment_width*.025))
            cuff_kernel=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,(2*cuff_radius+1,2*cuff_radius+1))
            # Source sleeve seams are often labelled as arm skin. Permit a
            # small cuff band even when the target is short-sleeved, while
            # keeping the rest of the forearm outside the editable region.
            cuff_band=(cv2.dilate(clothing,cuff_kernel)>0)&arms
            cuff_band &= sleeve_corridor(landmarks,person.size)>0
            sleeves |= cuff_band
            # Include the arm outline so a long sleeve can extend slightly
            # beyond bare skin. Keep the margin bounded by pose and scale.
            # Long sleeves need fabric volume outside a bare forearm. A skin-
            # tight margin leaves too little editable background for a cuff or
            # puffer sleeve and makes the network keep the source short sleeve.
            sleeve_radius = max(2,round(garment_width*(.06 if long_sleeves else .015)))
            sleeve_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,(2*sleeve_radius+1,2*sleeve_radius+1))
            sleeve_edges = cv2.dilate(sleeves.astype(np.uint8),sleeve_kernel)>0
            sleeve_edges &= sleeve_corridor(landmarks,person.size,long_sleeves=long_sleeves)>0
            if cropped_arms.any():
                cropped_edges=cv2.dilate(cropped_arms.astype(np.uint8),sleeve_kernel)>0
                outline=(np.asarray(foreground)>.45).astype(np.uint8)
                margin=max(2,round(garment_width*.02))
                outline=cv2.dilate(outline,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1)))>0
                sleeve_edges |= cropped_edges&outline
            mask[sleeve_edges & (labels==0)] = 255
            mask[sleeves] = 255
            if foreground is not None:
                # Preserve scene pixels outside a narrow band around the
                # person, even when the parser overestimates shirt edges.
                silhouette = (np.asarray(foreground)>.45).astype(np.uint8)
                body_radius=max(2,round(garment_width*.015))
                body_kernel=cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,(2*body_radius+1,2*body_radius+1))
                allowed_background = (cv2.dilate(silhouette,body_kernel)>0) | sleeve_edges | long_overlay
                protected |= (labels==0) & ~allowed_background
            rows = np.arange(person.height)[:, None]
            protected |= arms & ~sleeves & (rows > y0+(y1-y0)*.5)
            if not long_sleeves:
                protected |= arms & ~sleeves
        if category == 'lower':
            protected |= np.isin(labels, (5, 7, 14, 15))
            if short_bottoms:
                protected |= np.isin(labels,(16,17))
        if category == 'overall':
            # Arms/hands anchor pose. Leg labels are editable for a full outfit:
            # pale trousers are frequently misclassified as exposed legs, and
            # retaining those pixels leaks the old colour into the new outfit.
            # Shoes remain protected above; output joints are checked separately.
            arm = np.isin(labels,(14,15))
            # Let sleeves extend across the upper arm, while retaining the
            # forearm and hand that establish the person's pose.
            if landmarks is not None:
                arm &= upper_arm_editable(landmarks, person.size) == 0
            protected |= arm
            protected |= (labels==0)&~repair&~edge_repair
        if category in ('lower','overall'):
            foot = foot_protection(landmarks,person.size,labels)>0
            if category == 'overall':
                foot &= ~np.isin(labels,(9,12))
            protected |= foot
        hands = hand_protection(landmarks,person.size,clip_cuff=category=='upper' and long_sleeves)>0
        if category == 'upper':
            # A circular wrist guard also covers the old shirt cuff. Keep
            # actual hands and skin, but let recognised cuff fabric change.
            wrists = hand_protection(landmarks,person.size,protect_wrist=True,clip_cuff=long_sleeves)>0
            source_fabric = np.isin(labels,(5,6,7,10,11,9,12) if upper_length=='long' else (5,6,7,10,11))
            hands = (hands | wrists) & ~source_fabric
        protected |= hands
        if hem_limit is not None:
            below=np.arange(person.height)[:,None]>hem_limit
            protected |= below & (sleeve_corridor(landmarks,person.size)==0)
        mask[protected] = 0
        # Focus resolution on the garment while retaining head/body context.
        left, right = x0-garment_width*.25, x1+garment_width*.25
        top, bottom = y0-(y1-y0)*.15, y1+(y1-y0)*.15
        if category == 'upper' and sleeves.any():
            sleeve_y,sleeve_x = np.where(sleeves & (mask>0))
            if sleeve_x.size:
                left,right = min(left,sleeve_x.min()-radius),max(right,sleeve_x.max()+radius+1)
                top,bottom = min(top,sleeve_y.min()-radius),max(bottom,sleeve_y.max()+radius+1)
        if category in ('upper', 'overall'):
            head_y = np.where(np.isin(labels, (2, 13)))[0]
            if head_y.size: top = min(top, head_y.min()-garment_width*.08)
        if (category == 'upper' and silhouette is not None and not bare_torso.any()
                and landmarks is not None and len(landmarks) == 33
                and all(visible(landmarks[i],.5) for i in (0,11,12,23,24,27,28))):
            # Preserve the source build in the conditioning frame. A tight
            # torso crop loses both body proportions and sleeve context.
            # A shirtless source instead needs the focused crop above: in a
            # full frame, thin bare forearms collapse to a few latent pixels.
            left,right,top,bottom = 0,person.width,0,person.height
        # Include the complete editable area. Shrinking an oversized crop to
        # fit the image can clip sleeves or cuffs, especially in wide photos.
        edit_y,edit_x = np.where(mask>0)
        if edit_x.size:
            left,right = min(left,edit_x.min()),max(right,edit_x.max()+1)
            top,bottom = min(top,edit_y.min()),max(bottom,edit_y.max()+1)
        ratio = size[0]/size[1]
        crop_w, crop_h = right-left, bottom-top
        crop_w = max(crop_w, crop_h*ratio)
        crop_h = max(crop_h, crop_w/ratio)
        left = (left+right-crop_w)/2
        top = (top+bottom-crop_h)/2
        if crop_w <= person.width:
            left = max(0,min(left,person.width-crop_w))
        if crop_h <= person.height:
            top = max(0,min(top,person.height-crop_h))
        box = (round(left), round(top), round(left+crop_w), round(top+crop_h))
        feather = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(2))
        feather_array = np.array(feather)
        feather_array[protected] = 0
        feather = Image.fromarray(feather_array)
        crop = Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'white')
        crop.paste(person,(-box[0],-box[1]))
        prepared=(crop.resize(size, Image.Resampling.LANCZOS),
                  feather.crop(box).resize(size, Image.Resampling.BILINEAR), box)
        if include_protection:
            # Retain full-resolution protection; small fingers can disappear
            # during the 384x512 roundtrip otherwise.
            return (*prepared,Image.fromarray(protected.astype(np.uint8)*255))
        return prepared


def restore_crop(output, original, mask, box, protected=None):
    size = (box[2]-box[0], box[3]-box[1])
    result = original.copy()
    alpha = np.asarray(mask.resize(size, Image.Resampling.BILINEAR).convert('L'))
    # Soft mask edges otherwise blend the original garment colour into the result.
    alpha = np.where(alpha >= 96, 255, 0).astype(np.uint8)
    result.paste(output.resize(size, Image.Resampling.LANCZOS), box[:2], Image.fromarray(alpha))
    if protected is not None:
        result.paste(original,(0,0),protected.convert('L'))
    return result
