"""Train the local convolutional autoencoder on BSDS300."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from backend.models.learned_vae import LearnedVAE

def sample(files, size, rng):
    with Image.open(rng.choice(files)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if min(image.size) < size:
            scale = size / min(image.size); image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.LANCZOS)
        left, top = rng.randrange(image.width-size+1), rng.randrange(image.height-size+1)
        image = image.crop((left, top, left+size, top+size))
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()/255).permute(2,0,1)*2-1

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--validation',required=True); p.add_argument('--iterations',type=int,default=1000); p.add_argument('--crop-size',type=int,default=128); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--device',default='auto'); args=p.parse_args()
    train=sorted(x for x in Path(args.train).rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}); valid=sorted(x for x in Path(args.validation).rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if not train or not valid: raise SystemExit('Training and validation images are required.')
    device=torch.device('cuda' if args.device in {'auto','cuda','gpu'} and torch.cuda.is_available() else 'cpu'); rng=random.Random(42); vrng=random.Random(43)
    model=LearnedVAE(latent_channels=8).to(device).train(); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5); best=float('inf'); root=Path('models/autoencoder'); root.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.iterations+1):
        batch=torch.stack([sample(train,args.crop_size,rng) for _ in range(args.batch_size)]).to(device); out=model(batch); loss=F.l1_loss(out,batch)+0.25*F.mse_loss(out,batch)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%50==0 or step==args.iterations:
            model.eval()
            with torch.inference_mode(): val=torch.stack([sample(valid,args.crop_size,vrng) for _ in range(min(16,len(valid)))]).to(device); val_loss=float((F.l1_loss(model(val),val)+0.25*F.mse_loss(model(val),val)).item())
            model.train(); print(f'{step}/{args.iterations} train={loss.item():.5f} val={val_loss:.5f}',flush=True)
            blob={'model':model.state_dict(),'base_channels':64,'latent_channels':8,'iterations':step,'validation_loss':val_loss,'parameter_count':sum(x.numel() for x in model.parameters()),'codec_version':'LC3-8ch'}
            torch.save(blob,root/'latest_model.pth')
            if val_loss<best: best=val_loss; torch.save(blob,root/'best_model.pth'); (root/'metrics.json').write_text(json.dumps(blob | {'model':None},indent=2),encoding='utf-8')
    print(f'completed: {root / "best_model.pth"}')
if __name__=='__main__': main()
