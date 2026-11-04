"""Encode saved replay screenshots with Windows' existing H.264 encoder."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps


def main():
    dest=Path('outputs/rby1-research/stage23');files=sorted((dest/'media_frames').glob('frame_*.png'))
    check=json.loads((dest/'media_export_check.json').read_text());assert len(files)==check['screenshot_count']==71
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);frames=[]
    for file in files:
        image=Image.open(file).convert('RGB');w,h=image.size
        canvas=Image.new('RGB',(w+w%2,h+52+(h+52)%2),'white');canvas.paste(image,(0,52));draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'0.5倍速记录回放 | 左：原尺度机器人；右：下移放大示意',font=font,fill='#202020');frames.append(canvas)
    assert len({frame.size for frame in frames})==1
    output=dest/'grasp-stop-replay.mp4';fps=10
    writer=cv2.VideoWriter(str(output),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,frames[0].size)
    assert writer.isOpened(),'Windows H.264 encoder unavailable'
    playback=[frames[0]]*10+frames+[frames[-1]]*20
    try:
        for frame in playback:writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
    finally:writer.release()
    cap=cv2.VideoCapture(str(output));assert cap.isOpened();count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[1::-1]==frames[0].size;count+=1
    actual_fps=cap.get(cv2.CAP_PROP_FPS);codec=int(cap.get(cv2.CAP_PROP_FOURCC));cap.release()
    assert count==len(playback) and abs(actual_fps-fps)<.01
    gif_frames=[frame.quantize(colors=112) for frame in frames]
    gif_frames[0].save(dest/'急停后抓持漂移回放.gif',save_all=True,append_images=gif_frames[1:],duration=[1100]+[100]*69+[2100],loop=0,optimize=True,disposal=1)
    with Image.open(dest/'急停后抓持漂移回放.gif') as gif:assert gif.n_frames==71
    check.update({'mp4_file':str(output.resolve()),'mp4_bytes':output.stat().st_size,'mp4_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'mp4_decode_all_frames_passed':True,'encoded_frames':count,'fps':actual_fps,'duration_s':count/actual_fps,
        'codec_fourcc':''.join(chr((codec>>(8*i))&255) for i in range(4)),'dimensions_px':list(frames[0].size),
        'playback_speed':.5,'initial_hold_s':1.,'final_hold_s':2.,'gif_file':str((dest/'急停后抓持漂移回放.gif').resolve())})
    (dest/'media_export_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:check[k] for k in ['mp4_file','mp4_bytes','encoded_frames','duration_s','codec_fourcc','dimensions_px','offline_file_open_and_controls_passed']},ensure_ascii=False))


if __name__=='__main__':main()
